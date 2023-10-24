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


class DataCollatorForGenerativeClassification:  

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
        
        new_features = {'input_ids': [], 'label_mask': []}

        for feature in features:
            new_feature = {}

            new_feature['label_mask'] = torch.IntTensor([0]*feature['label_input_ids'].size(0) + [1] * feature['input_ids'].size(0) + [0] * (max_length - feature['input_ids'].size(0) - feature['label_input_ids'].size(0)))   
            # print(new_feature['label_mask'])
            # print(feature['input_ids'])
            new_feature['input_ids'] = torch.cat([feature['label_input_ids'], feature['input_ids']], dim=0)

            new_features['input_ids'].append(new_feature['input_ids'])
            new_features['label_mask'].append(new_feature['label_mask'])

        new_features['labels'] = torch.Tensor([feature['labels'] for feature in features])
        # for key, value in new_features.items():
        #     print(key, len(value))
        batch = self.tokenizer.pad(new_features, padding="longest").to(self.device)
        # print(batch)
        # input()
        # batch = {
        #             'input_ids': torch.cat(new_features['input_ids'], dim=0).to(self.device),
        #             'attention_mask': torch.cat(new_features['attention_mask'], dim=0).to(self.device),
        #             'label_mask': torch.cat(new_features['label_mask'], dim=0).to(self.device),
        #             'labels': labels.to(self.device)
        #         }

        return batch

def get_nll(model, tokenizer, batch, num_labels, num_labelstrings):
    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
    label_mask = batch['label_mask']
    logits = outputs.logits

    shift_logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1).contiguous()
    shift_target = batch['input_ids'][..., 1:].contiguous()

    nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))
    
    nll = nll * label_mask[..., 1:]
    
    deno = (shift_target.ne(tokenizer.pad_token_id).float() * label_mask[..., 1:]).sum(dim=-1)
    # input(deno)
    nll = nll.sum(dim=-1)/deno

    # print(nll)
    nll = nll.view(-1, num_labels, num_labelstrings)

    # print(nll)
    return nll

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
    data_collator = DataCollatorForGenerativeClassification(tokenizer=tokenizer, num_labels=num_labels, num_labelstrings=num_labelstrings, device=device)
    if not data_collator.tokenizer.pad_token:
        data_collator.tokenizer.pad_token_id = tokenizer.eos_token_id
    eval_dataloader = DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=args.batch_size*num_labels*num_labelstrings)

    print("starting evaluation now....")

    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
    fresults = open(args.results_file, "w")
    foutputs = open(args.outputs_file, "w")

    accurate = {
        'logsumexp': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'average': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'vote': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)]
        }
    all_predictions = {
        'logsumexp': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'average': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'vote': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)]
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
            labels = batch['labels'].view(-1, num_labels, num_labelstrings)[:, 0, 0].to(device) # just some trickery
            # print(labels)
            all_labels += labels.tolist()
            total += labels.size(0)

            if args.batch_by_labelstring:
                nlls = []
                for i in range(num_labels): # for each label
                    nlls_per_label = []
                    for j in range(num_labelstrings): # for each prompt per label
                        batch_per_labelstring = {k: v[i*sub_batch_size:(i+1)*sub_batch_size, ...] for k, v in batch.items()}
                        nlls_per_label.append(get_nll(model, tokenizer, batch_per_labelstring, 1, 1))
                        
                    nlls.append(torch.cat(nlls_per_label, dim=1))
                nll = torch.cat(nlls, dim=0)

            elif args.batch_by_label:
                nlls = []
                sub_batch_size = args.batch_size*num_labelstrings
                for i in range(num_labels):
                    batch_per_label = {k: v[i*sub_batch_size:(i+1)*sub_batch_size, ...] for k, v in batch.items()}
                    nlls.append(get_nll(model, tokenizer, batch_per_label, 1, num_labelstrings))
                nll = torch.cat(nlls, dim=0)
            else:
                nll = get_nll(model, tokenizer, batch, num_labels, num_labelstrings)
                
                if args.debug:
                    new_batch_text = tokenizer.batch_decode(batch['input_ids'])
                    for i in range(nll.size(0)):
                        r = num_labels * num_labelstrings
                        pprint(new_batch_text[i*r:(i+1)*r])
                        print(nll[i, :, :], labels[i])
                        input("debugging")
            
            for runid in range(args.num_runs):                
                for k in range(init_index, end_index):
                    if k < num_labelstrings:
                        ids = torch.from_numpy(np.random.choice(np.arange(num_labelstrings), k, replace=False)).to(device)
                        nll_subset = nll.index_select(dim=2, index=ids)
                    else:
                        nll_subset = nll
                    if args.debug:
                        print(labels)
                    
                    #logsumexp or arithmetic mean of probabilities
                    loss = -torch.logsumexp(-nll_subset, dim=2) + np.log(k)
                    result_logsumexp = loss.min(dim=1)[1]
                    if args.debug:
                        print(loss, result_logsumexp)

                    #average or geometric mean of probabilities
                    loss = torch.mean(nll_subset, dim=2)
                    result_average = loss.min(dim=1)[1]
                    if args.debug:
                        print(loss, result_average)

                    #harmonic mean of probabilities
                    loss = -np.log(k) + torch.logsumexp(nll_subset, dim=2)
                    result_vote = loss.min(dim=1)[1]

                    #vote
                    # result_vote = nll_subset.min(dim=1)[1].mode(dim=1)[0]
                    # print(nll_subset.min(dim=1)[1])
                    if args.debug:   
                        print(loss, result_vote)
                        input()

                    accurate['logsumexp'][runid][k-init_index] += result_logsumexp.eq(labels).int().sum().item()
                    accurate['average'][runid][k-init_index] += result_average.eq(labels).int().sum().item()
                    accurate['vote'][runid][k-init_index] += result_vote.eq(labels).int().sum().item()

                    all_predictions['logsumexp'][runid][k-init_index] += result_logsumexp.tolist()
                    all_predictions['average'][runid][k-init_index] += result_average.tolist()
                    all_predictions['vote'][runid][k-init_index] += result_vote.tolist()

           

    for runid in range(args.num_runs):
        result = { 
            "k": [],
            'accuracy_logsumexp': [],
            'accuracy_average': [],
            'accuracy_vote': [], 
            'confusion_matrix_logsumexp': [],
            'confusion_matrix_average': [],
            'confusion_matrix_vote': []
        }
        for k in range(init_index, end_index):
            result["k"].append(k)
            result['accuracy_logsumexp'].append(accurate['logsumexp'][runid][k-init_index]/total)
            result['accuracy_average'].append(accurate['average'][runid][k-init_index]/total)
            result['accuracy_vote'].append(accurate['vote'][runid][k-init_index]/total)
            
            result['confusion_matrix_logsumexp'].append(str(confusion_matrix(all_labels, all_predictions['logsumexp'][runid][k-init_index])))
            result['confusion_matrix_average'].append(str(confusion_matrix(all_labels, all_predictions['average'][runid][k-init_index])))
            result['confusion_matrix_vote'].append(str(confusion_matrix(all_labels, all_predictions['vote'][runid][k-init_index])))

            print(f"runid={runid}, k={k}, accuracy_logsumexp: {accurate['logsumexp'][runid][k-init_index]}/{total} or {accurate['logsumexp'][runid][k-init_index]/total}")
            print(f"confusion matrix: \n{confusion_matrix(all_labels, all_predictions['logsumexp'][runid][k-init_index])}")
        fresults.write(json.dumps(result) + "\n")
        
        outputfile = os.path.dirname(args.outputs_file) + f"/run-{runid}_" + os.path.basename(args.outputs_file)
        print(outputfile)
        with open(outputfile, "w") as foutputs:
            #print(len(all_predictions[runid]))
            #print(len(list(zip(*all_predictions[runid]))))
            predictions = [" ".join(map(str, item)) for item in zip(*all_predictions['logsumexp'][runid])]
            outputs = [f"{label} {output}" for label, output in zip(all_labels, predictions)]
            foutputs.write("\n".join(outputs) + "\n")

            

if __name__=="__main__":
    main()