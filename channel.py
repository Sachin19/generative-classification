import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from tqdm import tqdm
import json

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser, BitsAndBytesConfig
from datasets import load_dataset

# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

from config_channel_ablate import TASK2LABELSTRINGS as TASK2ABLATELABELSTRINGS
from config_channel import TASK2LABELSTRINGS, EXAMPLEFORMAT, EXAMPLEFORMAT_SPACE

from dataset_loaders import TASK2LOADER, TOKEN

import logging

def none_or_str(value):
    if value == "None" or value == "none":
        return None
    
    return value

@dataclass
class ScriptArguments:
    setting: Optional[str] = field(default="simple", metadata={"help": "ok"})
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
    batch_by_label: Optional[bool] = field(default=False, metadata={"help": "ok"})
    batch_by_labelstring: Optional[bool] = field(default=False, metadata={"help": "ok"})
    outputs_file: Optional[str] = field(default="sentence", metadata={"help": "ok"})
    results_file: Optional[str] = field(default="label", metadata={"help": "ok"})
    metric: Optional[str] = field(default="accuracy", metadata={"help": "ok"})
    model_dtype: Optional[str] = field(default="fp32", metadata={"help": "ok"})
    pmi: Optional[bool] = field(default=False, metadata={"help": "ok"})
    debug: Optional[bool] = field(default=False, metadata={"help": "ok"})
    device_map: Optional[bool] = field(default=False, metadata={"help": "ok"})
    text: Optional[bool] = field(default=False, metadata={"help": "ok"})
    bettertransformer: Optional[bool] = field(default=False, metadata={"help": "ok"})
    ablate_context: Optional[bool] = field(default=False, metadata={"help": "ok"})
    overwrite: Optional[bool] = field(default=False, metadata={"help": "rerun if results already exist"})

    num_runs: Optional[int] = field(default=5, metadata={"help": "ok"})
    jobid: Optional[int] = field(default=0, metadata={"help": "ok"})
    
    
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

if args.ablate_context:
    TASK2LABELSTRINGS = TASK2ABLATELABELSTRINGS

os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
os.makedirs(os.path.dirname(args.outputs_file), exist_ok=True)
logfile = os.path.dirname(args.outputs_file) + f"/{args.jobid}.log"
logging.basicConfig(filename=logfile, level=logging.DEBUG) #, encoding='utf-8'

def get_tokenized_dataset(raw_dataset, tokenizer, textfield="sentence", labelfield="label", label2id=None, space=False):
    def preprocess_function(examples):
        if space:
            x = tokenizer([EXAMPLEFORMAT_SPACE.format(text=example) for example in examples[textfield]], max_length=200, add_special_tokens=False, truncation=True)
        else:
            x = tokenizer([EXAMPLEFORMAT.format(text=example) for example in examples[textfield]], max_length=200, add_special_tokens=False, truncation=True)
        # x = tokenizer(["\""+example+"\"" for example in examples[textfield]], max_length=200, add_special_tokens=False, truncation=True)
        # x = tokenizer([example for example in examples[textfield]], max_length=200, add_special_tokens=False, truncation=True)
        if label2id is not None:
            x['labels'] = [label2id[label] for label in examples[labelfield]]
        else:
            x['labels'] = [label for label in examples[labelfield]]
        return x

    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    columns_to_remove = raw_dataset.column_names
    if label2id is None:
        columns_to_remove.remove(labelfield)
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    # logging.info(tokenized_dataset)
    tokenized_dataset.set_format("torch")

    return tokenized_dataset

def process_batch(batch, alllabelstrings_tokenized, i, j, device, tokenizer=None):
    labelstring_tokenized = alllabelstrings_tokenized[i][j]
    
    if args.debug:
        print(labelstring_tokenized)
        print(tokenizer.batch_decode(labelstring_tokenized.input_ids))
        input("labelstring")
    batch = {k: v.to(device) for k, v in batch.items()}
    batch_size, seq_len = batch['input_ids'].size()
    label_len = labelstring_tokenized['input_ids'].size(-1)
    
    expanded_batch_input_ids = batch['input_ids'].repeat_interleave(labelstring_tokenized['input_ids'].size(0), dim=0) # output size = (#labels*batch_size, L)
    expanded_label_input_ids = labelstring_tokenized['input_ids'].view(1, -1, label_len).expand(batch_size, -1, -1).contiguous().view(-1, label_len)
    input_ids = torch.cat([expanded_label_input_ids, expanded_batch_input_ids], dim=1)

    if args.debug:
        print(labelstring_tokenized)
        newstr = tokenizer.batch_decode(input_ids)
        print(newstr)
        print(input_ids)
        print(tokenizer(newstr).input_ids)
        input("labelstring")

    expanded_batch_attention_mask = batch['attention_mask'].repeat_interleave(labelstring_tokenized['attention_mask'].size(0), dim=0) # output size = (#labels*batch_size, L)
    expanded_label_attention_mask = labelstring_tokenized['attention_mask'].view(1, -1, label_len).expand(batch_size, -1, -1).contiguous().view(-1, label_len)
    attention_mask = torch.cat([expanded_label_attention_mask, expanded_batch_attention_mask], dim=1)
    
    label_mask = torch.ones_like(attention_mask)
    label_mask[:, :label_len] = 0

    batch['input_ids'] = input_ids
    batch['attention_mask'] = attention_mask

    batch.pop("labels", None)
    batch.pop('idx', None)

    return batch, label_mask


def merge_batches(batches, label_masks, tokenizer, device):
    maxlen = max([batch['input_ids'].size(-1) for batch in batches])
    batchsize = batches[0]['input_ids'].size(0)
    
    padmax = torch.empty(batchsize, maxlen).long().data.fill_(tokenizer.pad_token_id).to(device)
    maskmax = torch.ones_like(padmax).int().to(device)
   
    for ij in range(len(batches)):
        extralen = maxlen - batches[ij]['input_ids'].size(-1)
        batches[ij]['input_ids'] = torch.cat([batches[ij]['input_ids'], padmax[:, :extralen]], dim=-1)
        batches[ij]['attention_mask'] = torch.cat([batches[ij]['attention_mask'], 1-maskmax[:, :extralen]], dim=-1)
        label_masks[ij] = torch.cat([label_masks[ij], maskmax[:, :extralen]], dim=-1)
        
    batch = {'input_ids': torch.cat([batch['input_ids'] for batch in batches], dim=0), 'attention_mask': torch.cat([batch['attention_mask'] for batch in batches], dim=0)}
    label_mask = torch.cat(label_masks, dim=0)
    
    return batch, label_mask

def get_nll(model, tokenizer, batch, label_mask, num_labels, num_labelstrings):
    outputs = model(**batch)
    logits = outputs.logits

    shift_logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1).contiguous()
    shift_target = batch['input_ids'][..., 1:].contiguous()

    nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))
    
    nll = nll * label_mask[..., 1:]
    
    deno = (shift_target.ne(tokenizer.pad_token_id).float() * label_mask[..., 1:]).sum(dim=-1)
    # input(deno)
    nll = nll.sum(dim=-1)/deno
    # logging.info(nll)

    nll = nll.view(num_labels, num_labelstrings, -1)

    # logging.info(nll)
    return nll

def main():
    # print(TASK2LABELSTRINGS)
    try:
        with open(args.results_file) as fresults_exist:
            if len(fresults_exist.readlines()) >= args.num_runs and not args.overwrite:
                print(f'{args.results_file} already exists and is full. exiting.')
                logging.info(f'{args.results_file} already exists and is full. exiting.')
                return
    except Exception as e:
        pass 

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    task_items = args.task.split("#")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="models/", token=TOKEN)
    logging.info(tokenizer.eos_token)
    logging.info(tokenizer.bos_token)
    logging.info(tokenizer.pad_token)
    tokenizer.pad_token = tokenizer.eos_token    
    tokenizer.pad_token_id = tokenizer.eos_token_id

    ############
    if args.model_dtype == "bit4":
        model = AutoModelForCausalLM.from_pretrained(
                args.model, cache_dir="models/", trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                token=TOKEN,
                # use_flash_attention_2=True,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    #bnb_4bit_use_double_quant=True,
                    #bnb_4bit_quant_type='nf4'
                )
            )
        # model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", load_in_4bit=True, cache_dir="models/")
        device = model.device
        print(device)
    elif args.model_dtype == "bit8":
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", load_in_8bit=True, cache_dir="models/", token=TOKEN)
        device = model.device
    else:
        if args.model in ["EleutherAI/pythia-6.9b"]:
            args.device_map = True
        
        if not args.device_map:
            model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", token=TOKEN)#, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", device_map="auto", token=TOKEN)
        # model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-1b-redpajama-200b', trust_remote_code=True, attn_impl='triton')
        logging.info("model loaded on cpu")
        if args.model_dtype == "bf16":
            model.to(dtype=torch.bfloat16)
        elif args.model_dtype == "fp16":
            model.half()

        if args.bettertransformer:
            model = model.to_bettertransformer()

        if not args.device_map:
            model.to(device)
        logging.info("model loaded on gpu")

    model.eval()
    logging.info(f"{args.model} loaded")

    ## 
    logging.info(task_items[1:])
    suffix = "-text" if args.text else ""
    alllabelstrings = [[item.format(*task_items[1:]) for item in items] for items in TASK2LABELSTRINGS[task_items[0]+suffix]]
    # logging.info(alllabelstrings)
    # input()

    num_labels = len(alllabelstrings)
    num_labelstrings = len(alllabelstrings[0])

    if args.effective_batch_size is not None:
        args.batch_size = max(1, args.effective_batch_size // (num_labelstrings * num_labels))
        print(f"effective batch size: {args.effective_batch_size}, total labelstrings: {num_labelstrings * num_labels}, batch size: {args.batch_size}")

    alllabelstrings_tokenized = []
    for labelstrings in alllabelstrings:
        labelstrings_tokenized = []
        for labelstring in labelstrings:
            # logging.info(tokenizer(labelstring, add_special_tokens=False, padding=False, return_tensors="pt").to(device))
            # logging.info(tokenizer(labelstring, padding=False, return_tensors="pt").to(device))
            labelstrings_tokenized.append(tokenizer(labelstring, padding=False, return_tensors="pt").to(device))
        alllabelstrings_tokenized.append(labelstrings_tokenized)
    # input()
    ############

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

    # tokenized_dataset = get_tokenized_dataset(raw_dataset, "sentence", "label")
    label2id = None
    if args.label2id is not None:
        label2id = eval(args.label2id)
    tokenized_dataset = get_tokenized_dataset(raw_dataset, tokenizer, args.textfield, args.labelfield, label2id, ("gpt" in args.model or "pythia" in args.model or "opt" in args.model) and ("hate" in args.task))
    logging.info("datasets and tokenizer loaded")

    
    ##
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    data_collator.tokenizer.pad_token_id = tokenizer.eos_token_id
    eval_dataloader = DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=args.batch_size)

    logging.info("starting evaluation now....")

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
            subbatches = []
            label_masks = []
            labels = batch['labels'].to(device)
    
            all_labels += labels.tolist()
            total += labels.size(0)
            
            if args.batch_by_labelstring:
                nlls = []
                for i in range(len(alllabelstrings_tokenized)): # for each label
                    nlls_per_label = []
                    for j in range(num_labelstrings): # for each prompt per label
                        sub_batch, label_mask = process_batch(batch, alllabelstrings_tokenized, i, j, device, tokenizer)
                        
                        nlls_per_label.append(get_nll(model, tokenizer, sub_batch, label_mask, 1, 1))
                        # outputs_per_labelstring = model(**batch_per_label)
                        # logits1 = outputs_per_labelstring.logits

                        # shift_logprobs = torch.nn.functional.log_softmax(logits1[..., :-1, :], dim=-1).contiguous()
                        # shift_target = batch_per_label['input_ids'][..., 1:].contiguous()

                        # nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))
                        
                        # nll = nll * per_label_mask[..., 1:]
                        # deno = (shift_target.ne(tokenizer.pad_token_id).float() * per_label_mask[..., 1:]).sum(dim=-1)
                        # # input(deno)
                        # nll = nll.sum(dim=-1)/deno

                        # nlls_per_label.append(nll.view(1, 1, -1))
                    nlls.append(torch.cat(nlls_per_label, dim=1))
                nll = torch.cat(nlls, dim=0)

            else:
                for i in range(len(alllabelstrings_tokenized)): # for each label
                    for j in range(num_labelstrings): # for each prompt per label
                        sub_batch, label_mask = process_batch(batch, alllabelstrings_tokenized, i, j, device, tokenizer)
                        if args.debug:
                            print(tokenizer.batch_decode(sub_batch['input_ids']))
                            input("sub")
                        subbatches.append(sub_batch)
                        label_masks.append(label_mask)
                new_batch, label_mask = merge_batches(subbatches, label_masks, tokenizer, device)
                # print(new_batch['input_ids'].size())
                if args.batch_by_label:
                    nlls = []
                    sub_batch_size = new_batch['input_ids'].size(0)//num_labels #args.batch_size*num_labelstrings
                    for i in range(num_labels):
                        batch_per_label = {k: v[i*sub_batch_size:(i+1)*sub_batch_size, ...] for k, v in new_batch.items()}
                        per_label_mask = label_mask[i*sub_batch_size:(i+1)*sub_batch_size, ...]
                        # print(batch_per_label['input_ids'].size())
                        nlls.append(get_nll(model, tokenizer, batch_per_label, per_label_mask, 1, num_labelstrings))
                        # outputs_per_label = model(**batch_per_label)
                        # logits1 = outputs_per_label.logits

                        # shift_logprobs = torch.nn.functional.log_softmax(logits1[..., :-1, :], dim=-1).contiguous()
                        # shift_target = batch_per_label['input_ids'][..., 1:].contiguous()

                        # nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))
                        
                        # nll = nll * per_label_mask[..., 1:]
                        # deno = (shift_target.ne(tokenizer.pad_token_id).float() * per_label_mask[..., 1:]).sum(dim=-1)
                        # # input(deno)
                        # nll = nll.sum(dim=-1)/deno

                        # nlls.append(nll.view(1, num_labelstrings, -1))
                        # print(nlls[-1].size())
                    
                    nll = torch.cat(nlls, dim=0)
                else:
                    nll = get_nll(model, tokenizer, new_batch, label_mask, num_labels, num_labelstrings)
                    # logging.info("ok")
            
                if args.debug:
                    new_batch_text = tokenizer.batch_decode(new_batch['input_ids'])
                    for i in range(nll.size(2)):
                        r = num_labels * num_labelstrings
                        print(new_batch_text[i*r:(i+1)*r])
                        print(nll[:, :, i], labels[i])
                        input(f"debugging {i}")
                del new_batch
                del subbatches
                del label_masks
            
            for runid in range(args.num_runs):                
                for k in range(init_index, end_index):
                    if k < num_labelstrings:
                        ids = torch.from_numpy(np.random.choice(np.arange(num_labelstrings), k, replace=False)).to(device)
                        nll_subset = nll.index_select(dim=1, index=ids)
                    else:
                        nll_subset = nll
                    if args.debug:
                        print(labels)
                    
                    #logsumexp or arithmetic mean of probabilities
                    loss = -torch.logsumexp(-nll_subset, dim=1) + np.log(k)
                    result_logsumexp = loss.min(dim=0)[1]
                    if args.debug:
                        print(loss, result_logsumexp)

                    #average or geometric mean of probabilities
                    loss = torch.mean(nll_subset, dim=1)
                    result_average = loss.min(dim=0)[1]
                    if args.debug:
                        print(loss, result_average)

                    #harmonic mean of probabilities
                    loss = -np.log(k) + torch.logsumexp(nll_subset, dim=1)
                    result_vote = loss.min(dim=0)[1]

                    #vote
                    # result_vote = nll_subset.min(dim=0)[1].mode(dim=0)[0]
                    # logging.info(nll_subset.min(dim=0)[1])
                    if args.debug:   
                        logging.info(loss, result_vote)
                        input()

                    accurate['logsumexp'][runid][k-init_index] += result_logsumexp.eq(labels).int().sum().item()
                    accurate['average'][runid][k-init_index] += result_average.eq(labels).int().sum().item()
                    accurate['vote'][runid][k-init_index] += result_vote.eq(labels).int().sum().item()

                    all_predictions['logsumexp'][runid][k-init_index] += result_logsumexp.tolist()
                    all_predictions['average'][runid][k-init_index] += result_average.tolist()
                    all_predictions['vote'][runid][k-init_index] += result_vote.tolist()

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
                precision_per_class[i] = cm[i, i] / np.sum(cm[:, i])
                recall_per_class[i] = cm[i, i] / np.sum(cm[i, :])
            
            f1_score_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class)
            macro_average_f1_score = np.mean(f1_score_per_class)
            return macro_average_f1_score
        else:
            raise ValueError("Wrong metric")
        
    for runid in range(args.num_runs):
        result = { 
            "metric_name": args.metric,
            "k": [],
            'metric_geometric': [],
            'metric_arithmetic': [],
            'metric_harmonic': [], 
            'confusion_matrix_geometric': [],
            'confusion_matrix_arithmetic': [],
            'confusion_matrix_harmonic': []
        }
        for k in range(init_index, end_index):
            # print(sum(all_labels))
            cm_geometric = confusion_matrix(all_labels, all_predictions['average'][runid][k-init_index])
            cm_arithmetic = confusion_matrix(all_labels, all_predictions['logsumexp'][runid][k-init_index])
            cm_harmonic = confusion_matrix(all_labels, all_predictions['vote'][runid][k-init_index])

            result["k"].append(k)
            # result['accuracy_logsumexp'].append(accurate['logsumexp'][runid][k-init_index]/total)
            # result['accuracy_average'].append(accurate['average'][runid][k-init_index]/total)
            # result['accuracy_vote'].append(accurate['vote'][runid][k-init_index]/total)
            result['metric_geometric'].append(compute_metric(cm_geometric))
            result['metric_arithmetic'].append(compute_metric(cm_arithmetic))
            result['metric_harmonic'].append(compute_metric(cm_harmonic))
            
            result['confusion_matrix_geometric'].append(str(cm_geometric))
            result['confusion_matrix_arithmetic'].append(str(cm_arithmetic))
            result['confusion_matrix_harmonic'].append(str(cm_harmonic))

            logging.info(f"runid={runid}, k={k}, {args.metric}_arithmetic-mean: {result['metric_arithmetic'][-1]}")
            logging.info(f"confusion matrix: \n{cm_arithmetic}")
        fresults.write(json.dumps(result) + "\n")
        
        outputfile = os.path.dirname(args.outputs_file) + f"/run-{runid}_" + os.path.basename(args.outputs_file)
        logging.info(outputfile)
        with open(outputfile, "w") as foutputs:
            #logging.info(len(all_predictions[runid]))
            #logging.info(len(list(zip(*all_predictions[runid]))))
            predictions = [" ".join(map(str, item)) for item in zip(*all_predictions['logsumexp'][runid])]
            outputs = [f"{label} {output}" for label, output in zip(all_labels, predictions)]
            foutputs.write("\n".join(outputs) + "\n")

            

if __name__=="__main__":
    main()