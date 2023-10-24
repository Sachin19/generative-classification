import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from tqdm import tqdm
import json

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from datasets import load_dataset

from config import TASK2LABELSTRINGS
from dataset_loaders import TASK2LOADER

def none_or_str(value):
    if value == "None" or value == "none":
        return None
    
    return value

@dataclass
class ScriptArguments:
    task: Optional[str] = field(default="sentiment2", metadata={"help": "ok"})
    dataset: Optional[str] = field(default="glue", metadata={"help": "ok"})
    data_dir: Optional[none_or_str] = field(default="sst2", metadata={"help": "ok"})
    data_files: Optional[none_or_str] = field(default=None, metadata={"help": "ok"})
    split: Optional[none_or_str] = field(default="validation", metadata={"help": "ok"})
    model: Optional[str] = field(default="gpt2-large", metadata={"help": "ok"})
    textfield: Optional[str] = field(default="sentence", metadata={"help": "ok"})
    labelfield: Optional[str] = field(default="label", metadata={"help": "ok"})
    label2id: Optional[none_or_str] = field(default=None, metadata={"help": "ok"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "ok"})
    batch_by_label: Optional[bool] = field(default=False, metadata={"help": "ok"})
    batch_by_labelstring: Optional[bool] = field(default=False, metadata={"help": "ok"})
    outputs_file: Optional[str] = field(default="sentence", metadata={"help": "ok"})
    results_file: Optional[str] = field(default="label", metadata={"help": "ok"})
    model_dtype: Optional[str] = field(default="fp32", metadata={"help": "ok"})
    pmi: Optional[bool] = field(default=False, metadata={"help": "ok"})
    debug: Optional[bool] = field(default=False, metadata={"help": "ok"})
    device_map: Optional[bool] = field(default=False, metadata={"help": "ok"})
    bettertransformer: Optional[bool] = field(default=False, metadata={"help": "ok"})
    num_runs: Optional[int] = field(default=5, metadata={"help": "ok"})
    
    
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

def get_tokenized_dataset(raw_dataset, tokenizer):
    def preprocess_function(examples):
        # print(examples)
        text_input_ids = tokenizer([example for example in examples["text"]], padding=False, max_length=200, truncation=True).input_ids
        label_input_ids = tokenizer([example for example in examples["labelstring"]], padding=False, max_length=200, truncation=True).input_ids
        return_dict = {'label_input_ids': label_input_ids, 'input_ids': text_input_ids, 'labels': examples['labels']}
        return return_dict

    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    columns_to_remove = raw_dataset.column_names
    columns_to_remove.remove("labels")
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    print(tokenized_dataset)
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


class DataCollatorForDirectClassification:  

    def __init__(self, tokenizer, num_labels, num_labelstrings, device) -> None:
        self.tokenizer = tokenizer
        self.num_labels = num_labels
        self.num_labelstrings = num_labelstrings
        self.extra_padding = torch.empty(1, 200).int().data.fill_(tokenizer.pad_token_id)
        # self.extra_mask = torch.empty(label_features['input_ids'].size(0), 200).int().data.fill_(0).to(device)
        self.device = device


    def __call__(self, features):
        # def verify_batch(features):
            # return True
        # if not verify_batch(features):
            # raise ValueError("the batch_size should be a multiple of num_labels*num_labelstrings")

        # labels = torch.Tensor([feature['label'] for feature in features])
        max_length = max([feature['label_input_ids'].size(0) + feature['input_ids'].size(0) for feature in features])
        max_length_labelstring = max([feature['label_input_ids'].size(0) for feature in features])
        
        new_features = {'input_ids': [], 'label_mask': [], 'label_input_ids': [], 'label_attention_mask': []}

        for feature in features:
            new_feature = {}

            new_feature['label_mask'] = torch.IntTensor([0] * feature['input_ids'].size(0) + [1]*feature['label_input_ids'].size(0) + [0] * (max_length - feature['input_ids'].size(0) - feature['label_input_ids'].size(0)))   
            
            extra_padding_for_label_input_ids = torch.LongTensor([self.tokenizer.pad_token_id] * (max_length_labelstring - len(feature['label_input_ids'])))
            new_feature['label_attention_mask'] = torch.IntTensor([1] * feature['label_input_ids'].size(0) + [0]*extra_padding_for_label_input_ids.size(0))   
            new_feature['label_input_ids'] = torch.cat([feature['label_input_ids'], extra_padding_for_label_input_ids], dim=0)

            new_feature['input_ids'] = torch.cat([feature['input_ids'], feature['label_input_ids']], dim=0)

            new_features['input_ids'].append(new_feature['input_ids'])
            new_features['label_mask'].append(new_feature['label_mask'])
            new_features['label_input_ids'].append(new_feature['label_input_ids'])
            new_features['label_attention_mask'].append(new_feature['label_attention_mask'])

        new_features['labels'] = torch.Tensor([feature['labels'] for feature in features])
        batch = self.tokenizer.pad(new_features, padding="longest").to(self.device)

        return batch

def get_nll(model, tokenizer, batch, num_labels, num_labelstrings, pmi=True):
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    label_mask = batch['label_mask']
    logits = outputs.logits

    shift_logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1).contiguous()
    shift_target = batch['input_ids'][..., 1:].contiguous()

    nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))
    
    nll = nll * label_mask[..., 1:]
    
    deno = label_mask[..., 1:].sum(dim=-1)
    # input(deno)
    nll = nll.sum(dim=-1)/deno

    ##################
    outputs_ynull = model(input_ids=batch['label_input_ids'], attention_mask=batch['label_attention_mask'])
    logits = outputs_ynull.logits

    shift_logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1).contiguous()
    shift_target = batch['label_input_ids'][..., 1:].contiguous()

    nll_ynull = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))
    
    deno = shift_target.ne(tokenizer.pad_token_id).float().sum(dim=-1)
    # input(deno)
    nll_ynull = nll_ynull.sum(dim=-1)#/deno
    
    ###################

    # print(nll)
    nll = nll.view(-1, num_labels, num_labelstrings)
    nll_ynull = nll_ynull.view(-1, num_labels, num_labelstrings)

    # print(nll)
    return nll, nll_ynull

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    task_items = args.task.split("#")
    if task_items[0] in TASK2LOADER:
        # print(task_items)
        loader, params = TASK2LOADER[task_items[0]]
        params += task_items[1:]
        # print(params)
        # input()
        raw_dataset, num_labels, num_labelstrings = loader(*params)
    else:
        raise ValueError("Invalid task for this python script. Try channel.py or direct.py?")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="models/")

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token    
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized_dataset = get_tokenized_dataset(raw_dataset, tokenizer)
    print("datasets and tokenizer loaded")

    if args.model_dtype == "bit4":
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_4bit=True, cache_dir="models/")
        device = model.device
    elif args.model_dtype == "bit8":
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_8bit=True, cache_dir="models/")
        device = model.device
    else:
        if not args.device_map:
            model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/")#, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", device_map="auto")
        # model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-1b-redpajama-200b', trust_remote_code=True, attn_impl='triton')

        if args.model_dtype == "bf16":
            model.to(dtype=torch.bfloat16)
        elif args.model_dtype == "fp16":
            model.half()

        if args.bettertransformer:
            model = model.to_bettertransformer()

        model.to(device)

    model.eval()
    print(f"{args.model} loaded")

    ##
    data_collator = DataCollatorForDirectClassification(tokenizer=tokenizer, num_labels=num_labels, num_labelstrings=num_labelstrings, device=device)
    if not data_collator.tokenizer.pad_token:
        data_collator.tokenizer.pad_token_id = tokenizer.eos_token_id
    eval_dataloader = DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=args.batch_size*num_labels*num_labelstrings)

    print("starting evaluation now....")

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    fresults = open(args.results_file, "w")
    foutputs = open(args.outputs_file, "w")

    accurate = {
        'direct_logsumexp': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct_average': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct_vote': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_logsumexp': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_average': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_vote': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)]
        }
    all_predictions = {
        'direct_logsumexp': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct_average': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct_vote': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_logsumexp': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_average': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_vote': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)]
        }
    total = 0
    all_labels = []

    from pprint import pprint 
    init_index = 1
    if args.debug:
        init_index = num_labelstrings
    end_index = num_labelstrings+1
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            # print(batch['labels'].view(num_labels, num_labelstrings, -1))
            labels = batch['labels'].view(num_labels, num_labelstrings, -1)[0, 0].to(device) # just some trickery
            all_labels += labels.tolist()
            total += labels.size(0)

            if args.batch_by_labelstring:
                raise NotImplementedError

            elif args.batch_by_label:
                raise NotImplementedError
            
            else:
                nll, nll_ynull = get_nll(model, tokenizer, batch, num_labels, num_labelstrings)
                
                if args.debug:
                    new_batch_text = tokenizer.batch_decode(batch['input_ids'])
                    for i in range(nll.size(0)):
                        r = num_labels * num_labelstrings
                        pprint(new_batch_text[i*r:(i+1)*r])
                        print(nll[i, :, :], labels[i])
                        input("debugging")
            
            for prefix, additive in [("direct_", 0), ("direct++_", nll_ynull)]:
                nll = nll.view(-1, num_labels, num_labelstrings) - additive
                
                for runid in range(args.num_runs):
                    for k in range(1, num_labelstrings+1):
                        # print(f"k={k}", end=": ")
                        ids = torch.from_numpy(np.random.choice(np.arange(num_labelstrings), k, replace=False)).to(device)
                        nll_subset = nll.index_select(dim=2, index=ids)

                        # if args.aggregation == "logsumexp":
                        loss = -torch.logsumexp(-nll_subset, dim=2) + np.log(k)
                        result_logsumexp = loss.min(dim=1)[1]

                        # elif args.aggregation == "average":
                        loss = torch.mean(nll_subset, dim=2)
                        result_average = loss.min(dim=1)[1]

                        # else: #vote
                        result_vote = nll_subset.min(dim=1)[1].mode(dim=1)[0]

                        accurate[prefix+'logsumexp'][runid][k-1] += result_logsumexp.eq(labels).int().sum().item()
                        accurate[prefix+'average'][runid][k-1] += result_average.eq(labels).int().sum().item()
                        accurate[prefix+'vote'][runid][k-1] += result_vote.eq(labels).int().sum().item()

                        all_predictions[prefix+'logsumexp'][runid][k-1] += result_logsumexp.tolist()
                        all_predictions[prefix+'average'][runid][k-1] += result_average.tolist()
                        all_predictions[prefix+'vote'][runid][k-1] += result_vote.tolist()


    for runid in range(args.num_runs):
        result = { 
            "k": [],
            'direct_accuracy_logsumexp': [],
            'direct_accuracy_average': [],
            'direct_accuracy_vote': [], 
            'direct++_accuracy_logsumexp': [],
            'direct++_accuracy_average': [],
            'direct++_accuracy_vote': [], 
            'direct_confusion_matrix_logsumexp': [],
            'direct_confusion_matrix_average': [],
            'direct_confusion_matrix_vote': [],
            'direct++_confusion_matrix_logsumexp': [],
            'direct++_confusion_matrix_average': [],
            'direct++_confusion_matrix_vote': []
        }
        for k in range(1, num_labelstrings+1):
            result["k"].append(k)
            result['direct_accuracy_logsumexp'].append(accurate['direct_logsumexp'][runid][k-1]/total)
            result['direct_accuracy_average'].append(accurate['direct_average'][runid][k-1]/total)
            result['direct_accuracy_vote'].append(accurate['direct_vote'][runid][k-1]/total)

            result['direct++_accuracy_logsumexp'].append(accurate['direct++_logsumexp'][runid][k-1]/total)
            result['direct++_accuracy_average'].append(accurate['direct++_average'][runid][k-1]/total)
            result['direct++_accuracy_vote'].append(accurate['direct++_vote'][runid][k-1]/total)

            result['direct_confusion_matrix_logsumexp'].append(str(confusion_matrix(all_labels, all_predictions['direct_logsumexp'][runid][k-1])))
            result['direct_confusion_matrix_average'].append(str(confusion_matrix(all_labels, all_predictions['direct_average'][runid][k-1])))
            result['direct_confusion_matrix_vote'].append(str(confusion_matrix(all_labels, all_predictions['direct_vote'][runid][k-1])))

            result['direct++_confusion_matrix_logsumexp'].append(str(confusion_matrix(all_labels, all_predictions['direct++_logsumexp'][runid][k-1])))
            result['direct++_confusion_matrix_average'].append(str(confusion_matrix(all_labels, all_predictions['direct++_average'][runid][k-1])))
            result['direct++_confusion_matrix_vote'].append(str(confusion_matrix(all_labels, all_predictions['direct++_vote'][runid][k-1])))

            print(f"runid={runid}, k={k}, direct_accuracy_logsumexp: {accurate['direct_logsumexp'][runid][k-1]}/{total} or {accurate['direct_logsumexp'][runid][k-1]/total}")
            print(f"runid={runid}, k={k}, direct++_accuracy_logsumexp: {accurate['direct++_logsumexp'][runid][k-1]}/{total} or {accurate['direct++_logsumexp'][runid][k-1]/total}")
            
            print(f"direct confusion matrix: \n{confusion_matrix(all_labels, all_predictions['direct_logsumexp'][runid][k-1])}")
            print(f"direct++ confusion matrix: \n{confusion_matrix(all_labels, all_predictions['direct++_logsumexp'][runid][k-1])}")
        fresults.write(json.dumps(result) + "\n")
        
        outputfile = os.path.dirname(args.outputs_file) + f"/run-{runid}_" + os.path.basename(args.outputs_file)
        print(outputfile)
        with open(outputfile, "w") as foutputs:
            #print(len(all_predictions[runid]))
            #print(len(list(zip(*all_predictions[runid]))))
            predictions = [" ".join(map(str, item)) for item in zip(*all_predictions['direct_logsumexp'][runid])]
            outputs = [f"{label} {output}" for label, output in zip(all_labels, predictions)]
            foutputs.write("\n".join(outputs) + "\n")

            

if __name__=="__main__":
    main()