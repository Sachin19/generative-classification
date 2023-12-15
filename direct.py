import os
from dataclasses import dataclass, field
from typing import Literal, Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from tqdm import tqdm

import json
import logging

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

from datasets import load_dataset

from config_direct import TASK2LABELSTRINGS, TASK2SIMPLE, TASK2CONTEXT, TASK2INSTRUCT
from config_direct_ablate import TASK2LABELSTRINGS as TASK2ABLATELABELSTRINGS, TASK2SIMPLE as TASK2ABLATESIMPLE, TASK2CONTEXT as TASK2ABLATECONTEXT, TASK2INSTRUCT as TASK2ABLATEINSTRUCT
from dataset_loaders import TASK2LOADER, TOKEN

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def none_or_str(value):
    if value == "None" or value == "none":
        return None
    
    return value

@dataclass
class ScriptArguments:
    setting: Optional[str] = field(default="simple", metadata={"help": "ok"})
    perplexity_select: Optional[int] = field(default=None, metadata={"help": "ok"})
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
    effective_batch_size: Optional[int] = field(default=None, metadata={"help": "ok"})
    batch_by_label: Optional[bool] = field(default=False, metadata={"help": "compute forward pass per label separately"})
    outputs_file: Optional[str] = field(default="sentence", metadata={"help": "ok"})
    results_file: Optional[str] = field(default="label", metadata={"help": "ok"})
    metric: Optional[str] = field(default="accuracy", metadata={"help": "ok"})
    model_dtype: Optional[str] = field(default="fp32", metadata={"help": "ok"})
    pmi: Optional[bool] = field(default=False, metadata={"help": "ok"})
    text: Optional[bool] = field(default=False, metadata={"help": "ok"})
    device_map: Optional[bool] = field(default=False, metadata={"help": "ok"})
    bettertransformer: Optional[bool] = field(default=False, metadata={"help": "ok"})
    ablate_context: Optional[bool] = field(default=False, metadata={"help": "ok"})
    overwrite: Optional[bool] = field(default=False, metadata={"help": "ok"})


    num_runs: Optional[int] = field(default=5, metadata={"help": "ok"})
    jobid: Optional[int] = field(default=0, metadata={"help": "ok"})
    
    
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

if args.setting == "simple":
    TASK2PROMPT = TASK2SIMPLE
elif args.setting == "context":
    TASK2PROMPT = TASK2CONTEXT
elif args.setting == "instruct":
    TASK2PROMPT = TASK2INSTRUCT
else:
    raise ValueError(args.setting)

if args.ablate_context:
    TASK2LABELSTRINGS = TASK2ABLATELABELSTRINGS
    if args.setting == "simple":
        TASK2PROMPT = TASK2ABLATESIMPLE
    elif args.setting == "context":
        TASK2PROMPT = TASK2ABLATECONTEXT
    elif args.setting == "instruct":
        TASK2PROMPT = TASK2ABLATEINSTRUCT
    else:
        raise ValueError(args.setting)

os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
os.makedirs(os.path.dirname(args.outputs_file), exist_ok=True)

logfile = os.path.dirname(args.outputs_file) + f"/{args.jobid}.log"
print(f"logging in {logfile}")
logging.basicConfig(filename=logfile, level=logging.DEBUG)

class DataCollatorForDirectClassification:  

    def __init__(self, tokenizer, label_features, device) -> None:
        self.tokenizer = tokenizer
        self.label_features = label_features
        self.extra_padding = torch.empty(label_features['input_ids'].size(0), 200).int().data.fill_(tokenizer.pad_token_id).to(device)
        self.extra_mask = torch.empty(label_features['input_ids'].size(0), 200).int().data.fill_(0).to(device)
        self.device = device


    def __call__(self, features):
        labels = torch.Tensor([feature['label'] for feature in features])
        max_length = max([feature['input_ids'].size(0) + self.label_features['input_ids'].size(1) for feature in features])
        num_labelfeatures = self.label_features['input_ids'].size(0)
        
        new_features = {'input_ids': [], 'attention_mask': [], 'label_mask': []}

        for feature in features:
            new_feature = {}
            new_feature['input_ids'] = feature['input_ids'].view(1, -1).expand(num_labelfeatures, -1).to(self.device)

            extra_padding_len = max_length - self.label_features['input_ids'].size(1) - new_feature['input_ids'].size(1)
            new_feature['input_ids'] = torch.cat([new_feature['input_ids'], self.label_features['input_ids'], self.extra_padding[:, :extra_padding_len]], dim=1)

            new_feature['attention_mask'] = feature['attention_mask'].view(1, -1).expand(num_labelfeatures, -1).to(self.device)
            new_feature['label_mask'] = torch.zeros_like(new_feature['attention_mask']).int().to(self.device)

            extra_mask_len = max_length - self.label_features['attention_mask'].size(1) - new_feature['attention_mask'].size(1)
            new_feature['attention_mask'] = torch.cat([new_feature['attention_mask'], self.label_features['attention_mask'], self.extra_mask[:, :extra_mask_len]], dim=1)
            new_feature['label_mask'] = torch.cat([new_feature['label_mask'], self.label_features['attention_mask'], self.extra_mask[:, :extra_mask_len]], dim=1)

            new_features['input_ids'].append(new_feature['input_ids'])
            new_features['attention_mask'].append(new_feature['attention_mask'])
            new_features['label_mask'].append(new_feature['label_mask'])     

        batch = {
                    'input_ids': torch.cat(new_features['input_ids'], dim=0).to(self.device),
                    'attention_mask': torch.cat(new_features['attention_mask'], dim=0).to(self.device),
                    'label_mask': torch.cat(new_features['label_mask'], dim=0).to(self.device),
                    'labels': labels.to(self.device)
                }

        return batch

def get_tokenized_dataset(raw_dataset, tokenizer, textfield="sentence", labelfield="label", label2id=None):
    def preprocess_function(examples):
        # print(TASK2PROMPT.keys())
        x = tokenizer([TASK2PROMPT[args.task.split("#")[0]][0].format(text=example) for example in examples[textfield]], max_length=200, truncation=True)
        # x = tokenizer(examples[textfield], max_length=200, truncation=True)
        if label2id is not None:
                x['labels'] = [label2id[label] for label in examples[labelfield]]
        return x

    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    columns_to_remove = raw_dataset.column_names
    if label2id is None:
        columns_to_remove.remove(labelfield)
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    logging.info(tokenized_dataset)
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def get_nll(model, tokenizer, batch, label_mask):
    outputs = model(**batch)
    logits1 = outputs.logits
    
    shift_logprobs = torch.nn.functional.log_softmax(logits1[..., :-1, :], dim=-1).contiguous()
    shift_target = batch['input_ids'][..., 1:].contiguous()

    nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))
    # print("inside",nll)
    nll = nll * label_mask[..., 1:]
    # print(nll)
    # deno = label_mask[..., 1:].sum(dim=-1)
    nll = nll.sum(dim=-1) #/deno
    # print(nll.size())
    # input()
    # # deno = (shift_target.ne(tokenizer.pad_token_id).float() * label_mask[..., 1:]).sum(dim=-1)
    # # input(deno)
    # # nll = nll.sum(dim=-1)/deno
    # # logging.info(nll)

    # nll = nll.view(num_labels, num_labelstrings)

    # logging.info(nll)
    return nll

    


def main():
    try:
        print(args.results_file)
        with open(args.results_file, "r") as fresults_exist:
            l = len(fresults_exist.readlines())
            # print(l >= args.num_runs and not args.overwrite)
            # print(l)
            # print(not args.overwrite)
            # input()
            if l >= args.num_runs and not args.overwrite:
                print(f'{args.results_file} already exists and is full. exiting.')
                logging.info(f'{args.results_file} already exists and is full. exiting.')
                return
            else:
                print(f'{args.results_file} already exists but is incomplete, the code will run.')
    except Exception as e:
        # print(e)
        print(f'{args.results_file} does not exist, the code will run.')
        # input()
        pass 

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    task_items = args.task.split("#")
    if task_items[0] in TASK2LOADER:
        loader, params = TASK2LOADER[task_items[0]]
        params += task_items[1:]
        logging.info(params)
        raw_dataset = loader(*params)
        if isinstance(raw_dataset, tuple):
            raw_dataset, few_shot_data = raw_dataset 
    else:
        data_files=None
        if args.data_files is not None:
            data_files={args.split: args.data_files}
        raw_dataset = load_dataset(args.dataset, args.data_dir, split=args.split, data_files=data_files, cache_dir="datasets")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="models/", token=TOKEN)

    tokenizer.pad_token = tokenizer.eos_token    
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # tokenized_dataset = get_tokenized_dataset(raw_dataset, "sentence", "label")s
    label2id = None
    if args.label2id is not None:
        label2id = eval(args.label2id)
    tokenized_dataset = get_tokenized_dataset(raw_dataset, tokenizer, args.textfield, args.labelfield, label2id)
    print("datasets and tokenizer loaded")

    if args.model_dtype == "bit4":
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_4bit=True, cache_dir="models/", token=TOKEN)
        device = model.device
    elif args.model_dtype == "bit8":
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_8bit=True, cache_dir="models/", token=TOKEN)
        device = model.device
    else:
        if args.model in ["EleutherAI/pythia-6.9b"]:
            args.device_map = True

        if not args.device_map:
            model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", token=TOKEN)#, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", device_map="auto", token=TOKEN)
        # model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-1b-redpajama-200b', trust_remote_code=True, attn_impl='triton')

        if args.model_dtype == "bf16":
            model.to(dtype=torch.bfloat16)
        elif args.model_dtype == "fp16":
            model.half()

        if args.bettertransformer:
            model = model.to_bettertransformer()

        if not args.device_map:
            model.to(device)

    model.eval()
    print(f"{args.model} loaded")

    ## 
    logging.info(str(task_items[1:]))
    suffix = "-text" if args.text else ""
    alllabelstrings = [[item.format(*task_items[1:]) for item in items] for items in TASK2LABELSTRINGS[task_items[0]+suffix]]
    num_labels = len(alllabelstrings)
    num_labelstrings = len(alllabelstrings[0])

    if args.effective_batch_size is not None:
        args.batch_size = max(1, args.effective_batch_size // (num_labelstrings * num_labels))
        print(f"effective batch size: {args.effective_batch_size}, total labelstrings: {num_labelstrings * num_labels}, batch size: {args.batch_size}")

    if args.perplexity_select is None:
        args.perplexity_select = num_labelstrings
    
    flattened_labelstrings = []
    for labelstrings in alllabelstrings:
        flattened_labelstrings += labelstrings
    
    logging.info(len(flattened_labelstrings))
    alllabelstrings_tokenized = tokenizer(flattened_labelstrings, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
    alllabelstrings_tokenized_with_bos = tokenizer(flattened_labelstrings, padding=True, return_tensors="pt").to(device)
    #logging.info(alllabelstrings_tokenized)
    ##
    data_collator = DataCollatorForDirectClassification(tokenizer=tokenizer, label_features=alllabelstrings_tokenized, device=device)
    data_collator.tokenizer.pad_token_id = tokenizer.eos_token_id
    eval_dataloader = DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    logging.info("starting evaluation now....")

    fresults = open(args.results_file, "w")
    foutputs = open(args.outputs_file, "w")

    accurate = {
        'direct_arithmetic': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct_geometric': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct_harmonic': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_arithmetic': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_geometric': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_harmonic': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)]
        }
    all_predictions = {
        'direct_arithmetic': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct_geometric': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct_harmonic': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_arithmetic': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_geometric': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'direct++_harmonic': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)]
        }
    total = 0
    all_labels = []
    maxk = min(num_labelstrings, args.perplexity_select)

    with torch.no_grad():
        nll_ynull = 0
        if args.pmi:
            # logging.info(alllabelstrings_tokenized)
            outputs = model(**alllabelstrings_tokenized_with_bos)
            logits1 = outputs.logits
            
            shift_logprobs = torch.nn.functional.log_softmax(logits1[..., :-1, :], dim=-1).contiguous()
            shift_target = alllabelstrings_tokenized_with_bos['input_ids'][..., 1:].contiguous()

            nll_ynull = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))

            # deno = shift_target.ne(tokenizer.pad_token_id).float().sum(dim=-1)
            nll_ynull = nll_ynull.sum(dim=-1).view(1, num_labels, num_labelstrings)#/deno
            print(nll_ynull)

            if args.perplexity_select < num_labelstrings:
                description_indices = []
                nll_ynull_ = nll_ynull.view(num_labels, -1)
                logging.info(nll_ynull_)
                for labelid in range(num_labels):
                    nll_ynull_sorted = sorted(list(zip(list(range(len(nll_ynull_[labelid]))), nll_ynull_[labelid].tolist())), key=lambda x: x[1])
                    logging.info(nll_ynull_sorted)
                    description_indices.append(nll_ynull_sorted[0][0])

        for batch in tqdm(eval_dataloader):
            # logging.info(batch)
            # input()
            # subbatches = []
            # label_masks = []
            labels = batch['labels'].to(device)
            label_mask = batch['label_mask'].to(device)
            del batch['labels']
            del batch['label_mask']
    
            all_labels += labels.tolist()
            total += labels.size(0)
            
            
            if args.batch_by_label:
                nlls = []
                sub_batch_size = batch['input_ids'].size(0) // num_labels #args.batch_size*num_labelstrings
                for i in range(num_labels):
                    batch_per_label = {k: v[i*sub_batch_size:(i+1)*sub_batch_size, ...] for k, v in batch.items()}
                    per_label_mask = label_mask[i*sub_batch_size:(i+1)*sub_batch_size, ...]

                    nlls.append(get_nll(model, tokenizer, batch_per_label, per_label_mask))
                    
                nll = torch.cat(nlls, dim=0)
                # print(nll)
                # input()   
            else:
                outputs = model(**batch)
                logits1 = outputs.logits
                
                shift_logprobs = torch.nn.functional.log_softmax(logits1[..., :-1, :], dim=-1).contiguous()
                shift_target = batch['input_ids'][..., 1:].contiguous()

                nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))
                nll = nll * label_mask[..., 1:]
                # print(nll)

                #deno = label_mask[..., 1:].sum(dim=-1)
                nll = nll.sum(dim=-1) #/deno
                # print(nll)
                # input()
                
            for prefix, additive in [("direct_", 0), ("direct++_", nll_ynull)]:
                nll = nll.view(-1, num_labels, num_labelstrings) - additive
                
                for runid in range(args.num_runs):
                    for k in range(1, maxk+1):
                        # logging.info(f"k={k}", end=": ")
                        if args.perplexity_select == num_labelstrings:
                            ids = torch.from_numpy(np.random.choice(np.arange(num_labelstrings), k, replace=False)).to(device)
                        else:
                            ids = torch.LongTensor(description_indices[:maxk]).to(device)
                            # logging.info(ids)
                            # input()
                        nll_subset = nll.index_select(dim=2, index=ids)

                        # if args.aggregation == "logsumexp":
                        loss = -torch.logsumexp(-nll_subset, dim=2) + np.log(k)
                        result_logsumexp = loss.min(dim=1)[1]

                        # elif args.aggregation == "average":
                        loss = torch.mean(nll_subset, dim=2)
                        result_average = loss.min(dim=1)[1]

                        # else: #vote
                        # result_vote = nll_subset.min(dim=1)[1].mode(dim=1)[0]
                        #harmonic mean of probabilities
                        loss = -np.log(k) + torch.logsumexp(nll_subset, dim=1)
                        result_vote = loss.min(dim=1)[1]

                        accurate[prefix+'arithmetic'][runid][k-1] += result_logsumexp.eq(labels).int().sum().item()
                        accurate[prefix+'geometric'][runid][k-1] += result_average.eq(labels).int().sum().item()
                        accurate[prefix+'harmonic'][runid][k-1] += result_vote.eq(labels).int().sum().item()

                        all_predictions[prefix+'arithmetic'][runid][k-1] += result_logsumexp.tolist()
                        all_predictions[prefix+'geometric'][runid][k-1] += result_average.tolist()
                        all_predictions[prefix+'harmonic'][runid][k-1] += result_vote.tolist()

    def compute_metric(cm):
        if args.metric == "accuracy":
            cm = np.array(cm)
            num_classes = cm.shape[0]
            true_positives = np.sum(np.diag(cm))
            total_population = np.sum(cm)
            accuracy = true_positives / total_population
            return accuracy
        elif args.metric == "f1":
            
            cm = np.array(cm)
            num_classes = cm.shape[0]
        
            precision_per_class = np.zeros(num_classes)
            recall_per_class = np.zeros(num_classes)
            
            for i in range(num_classes):
                precision_per_class[i] = cm[i, i] / (np.sum(cm[:, i]) + 1e-7)
                recall_per_class[i] = cm[i, i] / (np.sum(cm[i, :]) + 1e-7)
            
            f1_score_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-7)
            macro_average_f1_score = np.mean(f1_score_per_class)
            return macro_average_f1_score
        else:
            raise ValueError("Wrong metric")

    for runid in range(args.num_runs):
        result = { 
            "metric_name": args.metric,
            "k": [],
            'direct_metric_arithmetic': [],
            'direct_metric_geometric': [],
            'direct_metric_harmonic': [], 
            'direct++_metric_arithmetic': [],
            'direct++_metric_geometric': [],
            'direct++_metric_harmonic': [], 
            'direct_confusion_matrix_arithmetic': [],
            'direct_confusion_matrix_geometric': [],
            'direct_confusion_matrix_harmonic': [],
            'direct++_confusion_matrix_arithmetic': [],
            'direct++_confusion_matrix_geometric': [],
            'direct++_confusion_matrix_harmonic': []
        }
        for k in range(1, maxk+1):
            cm_direct_arithmetic = confusion_matrix(all_labels, all_predictions['direct_arithmetic'][runid][k-1])
            cm_direct_geometric = confusion_matrix(all_labels, all_predictions['direct_geometric'][runid][k-1])
            cm_direct_harmonic = confusion_matrix(all_labels, all_predictions['direct_harmonic'][runid][k-1])

            cm_directpp_arithmetic = confusion_matrix(all_labels, all_predictions['direct++_arithmetic'][runid][k-1])
            cm_directpp_geometric = confusion_matrix(all_labels, all_predictions['direct++_geometric'][runid][k-1])
            cm_directpp_harmonic = confusion_matrix(all_labels, all_predictions['direct++_harmonic'][runid][k-1])

            result["k"].append(k)
            result['direct_metric_arithmetic'].append(compute_metric(cm_direct_arithmetic))
            result['direct_metric_geometric'].append(compute_metric(cm_direct_geometric))
            result['direct_metric_harmonic'].append(compute_metric(cm_direct_harmonic))

            result['direct++_metric_arithmetic'].append(compute_metric(cm_directpp_arithmetic))
            result['direct++_metric_geometric'].append(compute_metric(cm_directpp_geometric))
            result['direct++_metric_harmonic'].append(compute_metric(cm_directpp_harmonic))

            result['direct_confusion_matrix_arithmetic'].append(str(cm_direct_arithmetic))
            result['direct_confusion_matrix_geometric'].append(str(cm_direct_geometric))
            result['direct_confusion_matrix_harmonic'].append(str(cm_direct_harmonic))

            result['direct++_confusion_matrix_arithmetic'].append(str(cm_directpp_arithmetic))
            result['direct++_confusion_matrix_geometric'].append(str(cm_directpp_geometric))
            result['direct++_confusion_matrix_harmonic'].append(str(cm_directpp_harmonic))

            logging.info(f"runid={runid}, k={k}, direct_metric_arithmetic: {result['direct_metric_arithmetic'][-1]}")
            logging.info(f"runid={runid}, k={k}, direct_metric_arithmetic: {result['direct++_metric_arithmetic'][-1]}")
            
            logging.info(f"direct confusion matrix: \n{cm_direct_arithmetic}")
            logging.info(f"direct++ confusion matrix: \n{cm_directpp_arithmetic}")
        fresults.write(json.dumps(result) + "\n")
        
        outputfile = os.path.dirname(args.outputs_file) + f"/run-{runid}_" + os.path.basename(args.outputs_file)
        logging.info(outputfile)
        with open(outputfile, "w") as foutputs:
            #logging.info(len(all_predictions[runid]))
            #logging.info(len(list(zip(*all_predictions[runid]))))
            predictions = [" ".join(map(str, item)) for item in zip(*all_predictions['direct_arithmetic'][runid])]
            outputs = [f"{label} {output}" for label, output in zip(all_labels, predictions)]
            foutputs.write("\n".join(outputs) + "\n")

            

if __name__=="__main__":
    main()
