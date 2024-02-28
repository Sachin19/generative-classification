# change to contradict
# compare with other paper
# another model
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from tqdm import tqdm
import json
import re

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser, BitsAndBytesConfig, T5ForConditionalGeneration
from datasets import load_dataset, Dataset

# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

from config_channel_ablate import TASK2LABELSTRINGS as TASK2ABLATELABELSTRINGS
from config_custom import TASK2LABELSTRINGS#, EXAMPLEFORMAT2ENTAIL, EXAMPLEFORMAT2NOTENTAIL, EXAMPLEFORMAT_SPACE2ENTAIL, EXAMPLEFORMAT_SPACE2NOTENTAIL#, EXAMPLEFORMAT2, EXAMPLEFORMAT2_SPACE
from seeds import HF_SHUFFLE_SEED, FEWSHOT_SEED
from dataset_loaders import TASK2LOADER, TOKEN
from create_cat import create_cat
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
    textfield1: Optional[str] = field(default="sentence1", metadata={"help": "ok"})
    textfield2: Optional[str] = field(default="sentence2", metadata={"help": "ok"})
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
    hypGivenPrem: Optional[bool] = field(default=False, metadata={"help": "ok"})
    num_runs: Optional[int] = field(default=5, metadata={"help": "ok"})
    jobid: Optional[int] = field(default=0, metadata={"help": "ok"})
    cat: Optional[bool] = field(default=False, metadata={"help": "if this is for CAT calculation"})
    cat_seed: Optional[int] = field(default=1, metadata={"help": "seed for CAT"})
    
    
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]
num_labelstrings = 0
if args.ablate_context:
    TASK2LABELSTRINGS = TASK2ABLATELABELSTRINGS

os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
os.makedirs(os.path.dirname(args.outputs_file), exist_ok=True)
logfile = os.path.dirname(args.outputs_file) + f"/{args.jobid}.log"
logging.basicConfig(filename=logfile, level=logging.DEBUG) #, encoding='utf-8'
# TASK = "hGivenP_two_class"
class DataCollatorForNLI:  

    def __init__(self, tokenized_dataset, device) -> None: # what is label_features? I assume this is the examples
        self.tokenized_dataset = tokenized_dataset
        self.device = device


    def __call__(self, features): # tokenized inputs ->  
        labels = torch.Tensor([feature['labels'] for feature in features]) # get all labels from features

        new_features = {'input_ids': [], 'attention_mask': [], 'label_mask': [], 'decoder_input_ids': [], 'decoder_attention_mask': []} # we need three things 

        # print("^^^^^^^^^^^^^^^^")
        # print(features)
        for feature in features:
            new_features['input_ids'].append(feature['input_ids'])
            new_features['attention_mask'].append(feature['attention_mask'])
            new_features['label_mask'].append(feature['label_mask'])

            
        tmp = torch.stack(new_features['input_ids'], dim=0)
        
        n, num_class, num_label_strings, length = tmp.shape
        
        batch = { # make the dictionary
                    'input_ids': tmp.view(-1, length).to(self.device), #(B, num_class,num_labelstrings, length)
                    'attention_mask': torch.stack(new_features['attention_mask'], dim=0).view(-1, length).to(self.device),
                    'labels': labels.to(self.device),
                }
        batch['label_mask'] = torch.stack(new_features['label_mask'], dim=0).view(-1, length).to(self.device)
        
        return batch

def get_tokenized_dataset(raw_dataset, tokenizer, textfield1="sentence1", textfield2="sentence2", labelfield="label", label2id=None, space=False):
    def preprocess_function(examples): # would need to keep the prem and hyp separate, or just get the mask here
        pad_token_id = tokenizer.pad_token_id
        all_labelstrings = [lab for lab in TASK2LABELSTRINGS[args.task]] # (num classes by num_label_strings)
        n = len(examples['label'])
        num_labelstrings = len(all_labelstrings[0])
        num_classes = len(all_labelstrings)
        full_n = n * num_labelstrings
        if args.hypGivenPrem:
            masked_text, target_text = textfield1, textfield2
        else:
            target_text, masked_text = textfield1, textfield2
        tokens = np.full((n, num_classes, num_labelstrings), None)
        attention_masks = np.full((n, num_classes, num_labelstrings), None)
        label_masks = np.full((n, num_classes, num_labelstrings), None)
        
        tok_x = tokenizer.encode("x", add_special_tokens=False)

        for i in range(n):
            for j in range(num_classes):
                quoted_target = "\"" + examples[target_text][i] + "\""
                quoted_masked = "\"" + examples[masked_text][i] + "\""

                tokenized_target = tokenizer.encode("x " + quoted_target)[len(tok_x):]
                for k in range(num_labelstrings):
                    tmp_tok = tokenizer(all_labelstrings[j][k].format(text1=quoted_masked, text2=quoted_target))
                    tokens[i, j, k] = np.array(tmp_tok['input_ids'])
                    attention_masks[i, j, k] = np.array(tmp_tok['attention_mask'])
                    tmp_label_mask = np.zeros_like(tokens[i, j, k])
                    
                    idx = len(tokenized_target)
                    tmp_label_mask[-(idx):] = 1
                    label_masks[i, j, k] = tmp_label_mask


        max_len = 0
        for i in range(n):
            for j in range(num_classes):
                for k in range(num_labelstrings):
                    curr_length = len(tokens[i, j, k])
                    if curr_length > max_len:
                        max_len = curr_length

            

        padded_tokens = np.full((n, num_classes, num_labelstrings, max_len), pad_token_id)
        padded_attention_mask = np.full((n, num_classes, num_labelstrings, max_len), 0)
        padded_label_mask = np.full((n, num_classes, num_labelstrings, max_len), 0)

        for i in range(n):
            for j in range(num_classes):
                for k in range(num_labelstrings):
                    padded_tokens[i, j, k, :len(tokens[i, j, k])] = tokens[i, j, k]
                    padded_attention_mask[i, j, k, :len(attention_masks[i, j, k])] = attention_masks[i, j, k]
                    padded_label_mask[i, j, k, :len(label_masks[i, j, k])] = label_masks[i, j, k]
                    
        if label2id is not None:
            labels = np.array([label2id[label] for label in examples[labelfield]])
        else:
            labels = np.array([label for label in examples[labelfield]])

        tokenized_dataset = {
            'input_ids': torch.from_numpy(padded_tokens), # (n, num_classes, num_labelstrings, maxlen)
            'attention_mask': torch.from_numpy(padded_attention_mask), # (n, num_classes, num_labelstrings, maxlen)
            'labels': torch.from_numpy(labels), # (n)
            'label_mask': torch.from_numpy(padded_label_mask) # (n, num_classes, num_labelstrings, maxlen)
        }
            
        # print("***************")
        # print(tokenized_dataset['input_ids'].shape, tokenized_dataset['labels'].shape)
        # test_i = 0
        # test_j = 0
        # test_k = 0
        # s_tmp = all_labelstrings[test_j][test_k]
        # quoted_target = "\"" + examples[target_text][test_i] + "\""
        # quoted_masked = "\"" + examples[masked_text][test_i] + "\""
        # print(tokenizer(s_tmp.format(text1=quoted_masked, text2="")))
        # print("*****", padded_tokenized_targets[test_i, test_j, test_k])
        # print(padded_tokens[test_i, test_j, test_k, :])
        # print(padded_label_mask[test_i, test_j, test_k, :])
        # print(padded_attention_mask[test_i, test_j, test_k, :])
        # print(s_tmp.format(text1=quoted_masked, text2=quoted_target))
        # # print(splitted[0][3])
        # print(quoted_target)
        # print(tokenizer(quoted_target)["input_ids"])
        # print(padded_tokenized_targets[test_i, test_j, test_k])
        # print(quoted_masked)
        # print(tokenizer(quoted_masked, add_special_tokens=False)["input_ids"])
        # input()
        return tokenized_dataset
    
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    
    columns_to_remove = raw_dataset.column_names
    if label2id is None:
        columns_to_remove.remove(labelfield)
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    logging.info(tokenized_dataset)
    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def get_nll(model, tokenizer, batch, label_mask, num_labels, num_labelstrings):
    outputs = model(**batch)
    logits = outputs.logits
    # print(logits.shape)
    shift_logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1).contiguous()
    if args.model == "google/ul2":
        shift_target = batch['decoder_input_ids'][..., 1:].contiguous()
    else: 
        shift_target = batch['input_ids'][..., 1:].contiguous()
    # print(shift_target.shape)
    nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1)) # to ensure they correspond and pick the correct word from the dist. reduction is to not do anything extra

    nll = nll * label_mask[..., 1:]
    
    deno = (shift_target.ne(tokenizer.pad_token_id).float()).sum(dim=-1) # just to account for the lengths, not too important as lengths of x are the same
    # input(deno)
    nll = nll.sum(dim=-1)/deno # actual summation
    # logging.info(nll)
    # input()
    nll = nll.view(-1, num_labels, num_labelstrings) # reorganize this to (B, num_classes, num_labelstrings)

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
    # print("OK1")
    ############ create the model
    if args.model_dtype == "bit4":
        # print("bit4")
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
        # print(device)
    elif args.model_dtype == "bit8":
        # print("bit8")
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", load_in_8bit=True, cache_dir="models/", token=TOKEN)
        device = model.device
    else:
        # print("else")
        if args.model in ["EleutherAI/pythia-6.9b"]:
            args.device_map = True
        
        if not args.device_map:
            if args.model == "google/ul2":
                # print("HELLO!")
                model = T5ForConditionalGeneration.from_pretrained("google/ul2", cache_dir="models/", token=TOKEN)
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", token=TOKEN)#, device_map="auto")
        else:
            if args.model == "google/ul2":
                # print("HELLO")
                model = T5ForConditionalGeneration.from_pretrained("google/ul2", cache_dir="models/", device_map="auto", token=TOKEN)
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

    model.eval() # set on evaluation mode
    logging.info(f"{args.model} loaded")

    ## 
    logging.info(task_items[1:])
    # # print("________")
    # # print(args.text)
    suffix = "-text" if args.text else ""
    # alllabelstrings = [[item.format(*task_items[1:]) for item in items] for items in TASK2LABELSTRINGS[task_items[0]+suffix]]
    # # logging.info(alllabelstrings)
    # # input()
    # print(alllabelstrings) # these are the paraphrases
    num_labels = len(TASK2LABELSTRINGS[args.task])
    num_labelstrings = len(TASK2LABELSTRINGS[args.task][0]) # fix this TASK2LABELSTRINGS[args.task]

    if args.effective_batch_size is not None:
        args.batch_size = max(1, args.effective_batch_size // (num_labelstrings * num_labels))
        print(f"effective batch size: {args.effective_batch_size}, total labelstrings: {num_labelstrings * num_labels}, batch size: {args.batch_size}") # account for the paraphrases

    # alllabelstrings_tokenized = []
    # for labelstrings in alllabelstrings: # tokenize the paraphrases
    #     labelstrings_tokenized = []
    #     for labelstring in labelstrings:
    #         # logging.info(tokenizer(labelstring, add_special_tokens=False, padding=False, return_tensors="pt").to(device))
    #         # logging.info(tokenizer(labelstring, padding=False, return_tensors="pt").to(device))
    #         labelstrings_tokenized.append(tokenizer(labelstring, padding=False, return_tensors="pt").to(device))
    #     alllabelstrings_tokenized.append(labelstrings_tokenized)
    # # input()
    # ############
    print(TASK2LABELSTRINGS[args.task])
    if task_items[0] in TASK2LOADER: # not too important
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

        if args.cat:
            raw_dataset = create_cat(args.dataset, args.data_dir, split=args.split, text1=args.textfield1, text2=args.textfield2, labels=args.labelfield, seed=args.cat_seed)
        else:
            raw_dataset = load_dataset(args.dataset, args.data_dir, split=args.split, data_files=data_files, cache_dir="datasets")
        if len(raw_dataset) > 1000:
            raw_dataset = Dataset.from_dict(raw_dataset.shuffle(seed=HF_SHUFFLE_SEED)[:1000])

    # tokenized_dataset = get_tokenized_dataset(raw_dataset, "sentence", "label")
    label2id = None
    if args.label2id is not None:
        label2id = eval(args.label2id)
    # tokenize the premise and hypothesis
    tokenized_dataset = get_tokenized_dataset(raw_dataset, tokenizer, args.textfield1, args.textfield2, args.labelfield, label2id, ("gpt" in args.model or "pythia" in args.model or "opt" in args.model) and ("hate" in args.task))
    num_rows = tokenized_dataset.num_rows
    # print(tokenized_dataset['input_ids'].shape)
    # num_classes = tokenized_dataset['input_ids'].shape[1]
    # num_labelstrings = tokenized_dataset['input_ids'].shape[2]
    # print(hi)
    logging.info("datasets and tokenizer loaded")

    # # return None
    # ##
    # # padding?
    data_collator = DataCollatorForNLI(tokenized_dataset, device) # don't need to pass in tokenized_dataset
    # data_collator.tokenizer.pad_token_id = tokenizer.eos_token_id
    eval_dataloader = DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=args.batch_size) # it can iterate on its own

    logging.info("starting evaluation now....")

    fresults = open(args.results_file, "w")
    foutputs = open(args.outputs_file, "w")

    accurate = { 
        'logsumexp': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'average': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'vote': [[0 for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)]
        }
    # # print(accurate)
    all_predictions = {
        'logsumexp': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'average': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)],
        'vote': [[[] for k in range(1, num_labelstrings+1)] for runid in range(args.num_runs)]
        }
    # print("****")
    # print(len(all_predictions))
    total = 0
    all_labels = [] 

    from pprint import pprint 
    init_index = 1
    if args.debug:
        init_index = num_labelstrings
    end_index = num_labelstrings+1
    results = []
    nlls_ent = []
    nlls_not_ent = []
    with torch.no_grad():            
        for batch in tqdm(eval_dataloader):
            # print("&&&&&&&&&&&&&&&&&&&&")
            # print(batch)
            
            subbatches = []
            label_masks = []
            labels = batch['labels'].to(device)
            
            # print(batch['labels'])
            del batch['labels']
            # print(labels)
            if args.model != 'google/ul2':
                label_mask = batch['label_mask']
                del batch['label_mask']
            else:
                label_mask = None
            all_labels += labels.tolist()
            total += labels.size(0)
            # for key in batch.keys():
            #     print(key, batch[key].shape)
            # print(batch.keys())
            if args.batch_by_labelstring:
                nll = torch.empty(args.batch_size, num_labels, num_labelstrings).to(device)
                for i in range(num_labels): # for each label
                    for j in range(num_labelstrings): # for each prompt per label
                        sub_batch={}
                        # print(batch['input_ids'].shape)
                        sub_batch['input_ids'], sub_batch['attention_mask'], label_mask_ij = batch['input_ids'][:, i, j, :], batch['attention_mask'][: , i, j, :], label_mask[:, i, j, :]
                        val = get_nll(model, tokenizer, sub_batch, label_mask_ij, 1, 1)
                        nll[:, i, j] = val

            else:
                # print("YES!")
                nll = get_nll(model, tokenizer, batch, label_mask, num_labels, num_labelstrings)
                # print(nll.shape)
                    # logging.info("ok")
            
                if args.debug:
                    new_batch_text = tokenizer.batch_decode(batch['input_ids'])
                    for i in range(nll.size(2)):
                        r = num_labels * num_labelstrings
                        # print(new_batch_text[i*r:(i+1)*r])
                        # print(nll[:, :, i], labels[i])
                        input(f"debugging {i}")
                # del new_batch
                # del subbatches
                # del label_masks
            # print("@@@@@@@@@@@222")
            # print(labels)
            for runid in range(args.num_runs):  # we only compute probs once and compute mean and var by grouping them for the x-axis values              
                for k in range(init_index, end_index): # 0, 10 (we don't use 11 as there would be no variance, not ideal though due to the differences in 11 C k) 
                    # print("++++++++++++++++++")
                    # print(labels)
                    if k < num_labelstrings:
                        ids = torch.from_numpy(np.random.choice(np.arange(num_labelstrings), k, replace=False)).to(device)
                        # print(nll.shape)
                        # print(ids)
                        nll_subset = nll.index_select(dim=2, index=ids)
                    else:
                        nll_subset = nll
                    if args.debug:
                        print(labels)
                    # print()
                    #logsumexp or arithmetic mean of probabilities
                    loss = -torch.logsumexp(-nll_subset, dim=2) + np.log(k) # summing over nl probabilities. To sum, need to convert nll to ll, then exponentiate, them sum, then log again. To prevent underflow (batch size; no. labels).T
                    result_logsumexp = loss.min(dim=1)[1] # an array of label indices, i.e. the prediction
                    if args.debug:
                        print(loss, result_logsumexp)

                    #average or geometric mean of probabilities
                    loss = torch.mean(nll_subset, dim=2) # just different processing
                    result_average = loss.min(dim=1)[1]
                    if args.debug:
                        print(loss, result_average)

                    #harmonic mean of probabilities
                    loss = -np.log(k) + torch.logsumexp(nll_subset, dim=2)
                    result_vote = loss.min(dim=1)[1]

                    #vote
                    # result_vote = nll_subset.min(dim=0)[1].mode(dim=0)[0]
                    # logging.info(nll_subset.min(dim=0)[1])
                    if args.debug:   
                        logging.info(loss, result_vote)
                        input()
                    # print("$$$$$$$$$$$$$$$$")
                    # print(labels)
                    # print(result_logsumexp)
                    # print(accurate['logsumexp'])
                    # print(result_logsumexp.eq(labels))
                    accurate['logsumexp'][runid][k-init_index] += result_logsumexp.eq(labels).int().sum().item()
                    accurate['average'][runid][k-init_index] += result_average.eq(labels).int().sum().item()
                    accurate['vote'][runid][k-init_index] += result_vote.eq(labels).int().sum().item()

                    all_predictions['logsumexp'][runid][k-init_index] += result_logsumexp.tolist()
                    all_predictions['average'][runid][k-init_index] += result_average.tolist()
                    all_predictions['vote'][runid][k-init_index] += result_vote.tolist()

        
    def compute_metric(cm, metric):
        if metric == "accuracy":
            cm = np.array(cm)
            num_classes = cm.shape[0]
            true_positives = np.sum(np.diag(cm))
            total_population = np.sum(cm)
            accuracy = true_positives / total_population
            return accuracy
        elif metric == "f1":
            
            cm = np.array(cm)
            num_classes = cm.shape[0]
        
            precision_per_class = np.zeros(num_classes)
            recall_per_class = np.zeros(num_classes)
            
            for i in range(num_classes):
                precision_per_class[i] = cm[i, i] / np.sum(cm[:, i] + 1e-7)
                recall_per_class[i] = cm[i, i] / np.sum(cm[i, :] + 1e-7)
            
            f1_score_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-7)
            macro_average_f1_score = np.mean(f1_score_per_class)
            return macro_average_f1_score
        else:
            raise ValueError("Wrong metric")
        
    for runid in range(args.num_runs):
        result = { 
            # "metric_name": args.metric,
            "k": [],
            'f1_geometric': [],
            'f1_arithmetic': [],
            'f1_harmonic': [], 
            'accuracy_geometric': [],
            'accuracy_arithmetic': [],
            'accuracy_harmonic': [],
            'confusion_matrix_geometric': [],
            'confusion_matrix_arithmetic': [],
            'confusion_matrix_harmonic': []
        }
        for k in range(init_index, end_index):
            # print(sum(all_labels))
            cm_geometric = confusion_matrix(all_labels, all_predictions['average'][runid][k-init_index])
            cm_arithmetic = confusion_matrix(all_labels, all_predictions['logsumexp'][runid][k-init_index])
            cm_harmonic = confusion_matrix(all_labels, all_predictions['vote'][runid][k-init_index])
            # print(cm_arithmetic)
            # input()
            result["k"].append(k)
            # result['accuracy_logsumexp'].append(accurate['logsumexp'][runid][k-init_index]/total)
            # result['accuracy_average'].append(accurate['average'][runid][k-init_index]/total)
            # result['accuracy_vote'].append(accurate['vote'][runid][k-init_index]/total)
            for m in ["f1", "accuracy"]:
                result[m+'_geometric'].append(compute_metric(cm_geometric, m))
                result[m+'_arithmetic'].append(compute_metric(cm_arithmetic, m))
                result[m+'_harmonic'].append(compute_metric(cm_harmonic, m))
            
            result['confusion_matrix_geometric'].append(str(cm_geometric))
            result['confusion_matrix_arithmetic'].append(str(cm_arithmetic))
            result['confusion_matrix_harmonic'].append(str(cm_harmonic))

            # logging.info(f"runid={runid}, k={k}, {args.metric}_arithmetic-mean: {result['metric_arithmetic'][-1]}")
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

# pad_token_id = tokenizer.pad_token_id
# all_labelstrings = [lab for lab in TASK2LABELSTRINGS[args.task]] # (num classes by num_label_strings)
# n = len(examples['label'])
# prem_idx = 1
# hyp_idx = 3
# pred_idx = 3
# text_idx = [prem_idx, hyp_idx]
# textfields = [textfield1, textfield2]
# other_idx = [0, 2] # the non prem and hyp indices
# num_idx = 4
# num_labelstrings = len(all_labelstrings[0])
# num_classes = len(all_labelstrings)
# full_n = n * num_labelstrings

# splitted = []

# for i in range(num_classes): #re.split(r'({.*?})', labelstring)
#     splitted.append(np.array([labelstring.split("<break>") for labelstring in all_labelstrings[i]])) # This premise | {text1} |entailts this: | {text2}
# print(splitted[0][0])
# # print("*******", tokenitokenizer.bos_token)
# # print("&*************")
# # print(examples[textfield1][0])
# # print(examples[textfield2][0])
# # print(examples['label'][0])
# # print(splitted[0][0])
# # print(splitted[1][0])
# # print(q)
# tokens = np.full((n, num_classes, num_labelstrings, num_idx), None)
# attention_masks = np.full((n, num_classes, num_labelstrings, num_idx), None)
# # print(tokenizer(" ", add_special_tokens=False))
# # print(tokenizer(" \""))
# # print()
# # this premise: + "prem" + entails this hypothesis + "hyp"
# # this premise: " + prem + " entails this hypothesis " + hyp"
# print("(((((((((((((((())))))))))))))))")
# print(tokenizer(tokenizer.bos_token))
# print(tokenizer.decode([1]))
# print("("+ tokenizer.decode([28723,   345])+ ")")
# print("("+ tokenizer.decode([345, 1014, 18946])+")") # check if the model combines space and quotation, and use sliding window?
# print(len(tokenizer(" \"", add_special_tokens=False)["input_ids"]))
# print(tokenizer("\"", add_special_tokens=False))
# print(tokenizer(" \"", add_special_tokens=True))
# print(tokenizer(" \"", add_special_tokens=False))
# # input()
# for i in range(num_classes): # which class
#     for j in other_idx: # which index we want to tokenize
#         # if args.model=="meta-llama/Llama-2-13b-hf" or args.model=="huggyllama/llama-7b" or args.model =="meta-llama/Llama-2-7b-hf":
#         #     bos_token = tokenizer(tokenizer.bos_token)
#         #     if j == 0:
#         #         tok_i_j = np.array([np.array(tokenizer(s.strip(), add_special_tokens=False)['input_ids']) for s in splitted[i][:, j]])
#         #         att_mask_i_j = np.array([np.array(tokenizer(s.strip(), add_special_tokens=False)['attention_mask']) for s in splitted[i][:, j]])
                
#         #     else:
#         #         tok_i_j = np.array([np.array(tokenizer(s.strip(), add_special_tokens=False)['input_ids']) for s in splitted[i][:, j]])
#         #         att_mask_i_j = np.array([np.array(tokenizer(s.strip(), add_special_tokens=False)['attention_mask']) for s in splitted[i][:, j]])
#         # else:
#         if j != 0:
#             tok_i_j = np.array([np.array(tokenizer(s, add_special_tokens=False)['input_ids']) for s in splitted[i][:, j]])
#             att_mask_i_j = np.array([np.array(tokenizer(s, add_special_tokens=False)['attention_mask']) for s in splitted[i][:, j]])
#         else:
#             tok_i_j = np.array([np.array(tokenizer(s, add_special_tokens=True)['input_ids']) for s in splitted[i][:, j]])
#             att_mask_i_j = np.array([np.array(tokenizer(s, add_special_tokens=True)['attention_mask']) for s in splitted[i][:, j]])
#         tokens[:,i, :, j] = np.expand_dims(tok_i_j, 0)
    
#         attention_masks[:, i, :, j] = np.expand_dims(att_mask_i_j, 0)
# print(tokens[0, 1, 0, 1])
# for i, textfield in zip(text_idx, textfields):
#     # if args.model=="meta-llama/Llama-2-13b-hf" or args.model=="huggyllama/llama-7b" or args.model =="meta-llama/Llama-2-7b-hf":
#     #     input_ids_i = np.array([np.array(tokenizer("\""+example.strip()+"\"", add_special_tokens=False)['input_ids']) for example in examples[textfield]]) # 277 long
#     #     att_mask_i = np.array([np.array(tokenizer("\""+example.strip()+"\"", add_special_tokens=False)['attention_mask']) for example in examples[textfield]]) # 277 long
#     # else:
#     input_ids_i = np.array([np.array(tokenizer(example, add_special_tokens=False)['input_ids']) for example in examples[textfield]]) # 277 long
#     att_mask_i = np.array([np.array(tokenizer(example, add_special_tokens=False)['attention_mask']) for example in examples[textfield]]) # 277 long
#     tokens[:, :, :, i] = np.expand_dims(input_ids_i, (1, 2))
#     attention_masks[:, :, :, i] = np.expand_dims(att_mask_i, (1, 2))

# max_len = 0
# for i in range(n):
#     for j in range(num_classes):
#         for k in range(num_labelstrings):
#             curr_length = 0
#             for l in range(num_idx):
#                 curr_length += len(tokens[i, j, k, l])
#             if curr_length > max_len:
#                 max_len = curr_length


# label_masks = np.full(tokens.shape, None)
# for i in range(n):
#     for j in range(num_classes):
#         for k in range(num_labelstrings):
#             for l in range(num_idx):
#                 if l != pred_idx:
#                     label_masks[i, j, k, l] = np.zeros_like(tokens[i, j, k, l])
#                 else:
#                     label_masks[i, j, k, l] = np.ones_like(tokens[i, j, k, l])

# concatted_tokens = np.full((n, num_classes, num_labelstrings, max_len), pad_token_id)
# concatted_attn_mask = np.full((n, num_classes, num_labelstrings, max_len), 0)
# concatted_label_mask = np.full((n, num_classes, num_labelstrings, max_len), 0)

# for i in range(n):
#     for j in range(num_classes):
#         for k in range(num_labelstrings):
#             tmp_tok = np.concatenate(tokens[i, j, k, :])
#             tmp_attn = np.concatenate(attention_masks[i, j, k, :])
#             tmp_label = np.concatenate(label_masks[i, j, k, :])

#             concatted_tokens[i, j, k, :len(tmp_tok)] = tmp_tok
#             concatted_attn_mask[i, j, k, :len(tmp_tok)] = tmp_attn
#             concatted_label_mask[i, j, k, :len(tmp_tok)] = tmp_label

# labels = None
# if label2id is not None:
#     labels = np.array([label2id[label] for label in examples[labelfield]])
# else:
#     labels = np.array([label for label in examples[labelfield]])
# s_tmp = "This: \"{text1}\" contradicts this: \"{text2}\""
# print(tokenizer(s_tmp.format(text1=examples[textfield1][0], text2=examples[textfield2][0])))
# print(concatted_tokens[0, 0, 0, :])
# print(s_tmp.format(text1=examples[textfield1][0], text2=examples[textfield2][0]))
# print(splitted[0][3])
# print(tokenizer("\"" + examples[textfield1][0] + "\"", add_special_tokens=False)["input_ids"])
# print(tokenizer("\"" + examples[textfield2][0] + "\"", add_special_tokens=False)["input_ids"])
# tokenized_dataset = {
#     'input_ids': torch.from_numpy(concatted_tokens), # (n, num_classes, num_labelstrings, maxlen)
#     'attention_mask': torch.from_numpy(concatted_attn_mask), # (n, num_classes, num_labelstrings, maxlen)
#     'label_mask': torch.from_numpy(concatted_label_mask), # (n, num_classes, num_labelstrings, maxlen)
#     'labels': torch.from_numpy(labels) # (n)
# }

# # for key in tokenized_dataset.keys():
# #     print(f"{key}: {tokenized_dataset[key].shape}")

# # for key in tokenized_dataset.keys():
# #     if key != 'labels':
# #         print(tokenized_dataset[key][:2, :, :, :])

# return tokenized_dataset

if __name__=="__main__":
    main()  



# pad_token_id = tokenizer.pad_token_id
#         all_labelstrings = [lab for lab in TASK2LABELSTRINGS[args.task]] # (num classes by num_label_strings)
#         n = len(examples['label'])
#         num_labelstrings = len(all_labelstrings[0])
#         num_classes = len(all_labelstrings)
#         full_n = n * num_labelstrings
#         target_text = textfield2
#         # for each  class
#         # for each label string
#         # for each example
#         # tokenize and find length of example to pad.

#         tokens = np.full((n, num_classes, num_labelstrings), None)
#         attention_masks = np.full((n, num_classes, num_labelstrings), None)
#         label_masks = np.full((n, num_classes, num_labelstrings), None)

#         for i in range(n):
#             for j in range(num_classes):
#                 for k in range(num_labelstrings):
#                     quoted_tfield1 = "\"" + examples[textfield1][i] + "\""
#                     quoted_tfield2 = "\"" + examples[textfield2][i] + "\""
#                     tmp_tok = tokenizer(all_labelstrings[j][k].format(text1=quoted_tfield1, text2=quoted_tfield2))
#                     tokens[i, j, k] = np.array(tmp_tok['input_ids'])
#                     attention_masks[i, j, k] = np.array(tmp_tok['input_ids'])
#                     tmp_label_mask = np.zeros_like(tokens[i, j, k])
#                     tmp_label_mask[-len(tokenizer(quoted_tfield2)['input_ids']):] = 1
#                     label_masks[i, j, k] = tmp_label_mask

#         max_len = 0
#         for i in range(n):
#             for j in range(num_classes):
#                 for k in range(num_labelstrings):
#                     curr_length = len(tokens[i, j, k])
#                     if curr_length > max_len:
#                         max_len = curr_length
        
#         padded_tokens = np.full((n, num_classes, num_labelstrings, max_len), pad_token_id)
#         padded_attention_mask = np.full((n, num_classes, num_labelstrings, max_len), 0)
#         padded_label_mask = np.full((n, num_classes, num_labelstrings, max_len), 0)

#         for i in range(n):
#             for j in range(num_classes):
#                 for k in range(num_labelstrings):
#                     padded_tokens[i, j, k, :len(tokens[i, j, k])] = tokens[i, j, k]
#                     padded_attention_mask[i, j, k, :len(attention_masks[i, j, k])] = attention_masks[i, j, k]
#                     padded_label_mask[i, j, k, :len(label_masks[i, j, k])] = label_masks[i, j, k]
    
#         if label2id is not None:
#             labels = np.array([label2id[label] for label in examples[labelfield]])
#         else:
#             labels = np.array([label for label in examples[labelfield]])

#         tokenized_dataset = {
#             'input_ids': torch.from_numpy(padded_tokens), # (n, num_classes, num_labelstrings, maxlen)
#             'attention_mask': torch.from_numpy(padded_attention_mask), # (n, num_classes, num_labelstrings, maxlen)
#             'label_mask': torch.from_numpy(padded_label_mask), # (n, num_classes, num_labelstrings, maxlen)
#             'labels': torch.from_numpy(labels) # (n)
#         }
# s_tmp = "This: \"{text1}\" contradicts this: \"{text2}\""
#         print(tokenizer(s_tmp.format(text1=examples[textfield1][0], text2=examples[textfield2][0])))
#         print(padded_tokens[0, 0, 0, :])
#         print(padded_label_mask[0, 0, 0, :])
#         print(padded_attention_mask[0, 0, 0, :])
#         print(s_tmp.format(text1=examples[textfield1][0], text2=examples[textfield2][0]))
#         # print(splitted[0][3])
#         print(tokenizer("\"" + examples[textfield1][0] + "\"", add_special_tokens=False)["input_ids"])
#         print(tokenizer("\"" + examples[textfield2][0] + "\"", add_special_tokens=False)["input_ids"])

#         return tokenized_dataset