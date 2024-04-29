import sys
print("*************")
# print(sys.path)
unwanted_path = '/mmfs1/home/hsethu/.local/lib/python3.9/site-packages'
if unwanted_path in sys.path:
    sys.path.remove(unwanted_path)

import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from tqdm import tqdm
import json
import re
import hf_olmo
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import linregress

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser, BitsAndBytesConfig, T5ForConditionalGeneration
from datasets import load_dataset, Dataset

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

from config_channel_ablate import TASK2LABELSTRINGS as TASK2ABLATELABELSTRINGS
from config_fewshot_new import TASK2LABELSTRINGS, EXAMPLEFORMAT2ENTAIL, EXAMPLEFORMAT2NOTENTAIL, EXAMPLEFORMAT_SPACE2ENTAIL, EXAMPLEFORMAT_SPACE2NOTENTAIL#, EXAMPLEFORMAT2, EXAMPLEFORMAT2_SPACE

from calibration import cc
from dataset_loaders import TASK2LOADER, TOKEN
from nli_fewshot_balanced import get_tokenized_dataset_nli_fewshot
from nli_fewshot_custom_imbalance import get_tokenized_dataset_nli_fewshot_custom
from mcq_diff_fewshot import get_tokenized_dataset_mcq_diff_fewshot
from mcq_fewshot import get_tokenized_dataset_mcq_fewshot
from mcq_context_fewshot import get_tokenized_dataset_mcq_context_fewshot
# from cat_nli_fewshot import get_tokenized_dataset_nli_fewshot_cat
from create_cat import create_cat
from get_mmlu import get_mmlu
import logging
from seeds import HF_SHUFFLE_SEED, FEWSHOT_SEED

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
    label_names: Optional[str] = field(default="", metadata={"help": "ok"})
    type_of_task: Optional[str] = field(default="", metadata={"help": "ok"})
    question: Optional[str] = field(default="premise", metadata={"help": "ok"})
    choices: Optional[str] = field(default="", metadata={"help": "ok"})
    labelfield: Optional[str] = field(default="label", metadata={"help": "ok"})
    label2id: Optional[none_or_str] = field(default=None, metadata={"help": "ok"})
    possible_labels: Optional[str] = field(default="", metadata={"help": "ok"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "ok"})
    effective_batch_size: Optional[int] = field(default=None, metadata={"help": "ok"})
    batch_by_label: Optional[bool] = field(default=False, metadata={"help": "ok"})
    batch_by_labelstring: Optional[bool] = field(default=True, metadata={"help": "ok"})
    outputs_file: Optional[str] = field(default="sentence", metadata={"help": "ok"})
    results_file: Optional[str] = field(default="label", metadata={"help": "ok"})
    cc_outputs_file: Optional[str] = field(default="sentence", metadata={"help": "ok"})
    metric: Optional[str] = field(default="accuracy", metadata={"help": "ok"})
    model_dtype: Optional[str] = field(default="fp32", metadata={"help": "ok"})
    pmi: Optional[bool] = field(default=False, metadata={"help": "ok"})
    debug: Optional[bool] = field(default=False, metadata={"help": "ok"})
    device_map: Optional[bool] = field(default=False, metadata={"help": "ok"})
    text: Optional[bool] = field(default=False, metadata={"help": "ok"})
    bettertransformer: Optional[bool] = field(default=False, metadata={"help": "ok"})
    ablate_context: Optional[bool] = field(default=False, metadata={"help": "ok"})
    overwrite: Optional[bool] = field(default=False, metadata={"help": "rerun if results already exist"})
    aGivenQ: Optional[bool] = field(default=False, metadata={"help": "ok"})
    num_runs: Optional[int] = field(default=5, metadata={"help": "ok"})
    jobid: Optional[int] = field(default=0, metadata={"help": "ok"})
    k: Optional[int] = field(default=3, metadata={"help": "ok"}) # num few shot examples
    num_sets: Optional[int] = field(default=4, metadata={"help": "ok"}) # num sets of examples
    num_fewshot_perms: Optional[int] = field(default=1, metadata={"help": "ok"}) # useless
    passage: Optional[str] = field(default="", metadata={"help": "ok"}) 
    cat: Optional[bool] = field(default=False, metadata={"help": "if this is for CAT calculation"})
    cat_seed: Optional[int] = field(default=1, metadata={"help": "seed for generating cat"})
    want_choice: Optional[bool] = field(default=False, metadata={"help": "ok"})
    _lambda: Optional[int] = field(default=10, metadata={"help": "num layers to use for the slope"})
    
  

parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]
total_perm_sets = 0
if args.ablate_context:
    TASK2LABELSTRINGS = TASK2ABLATELABELSTRINGS

os.makedirs(os.path.dirname(args.results_file), exist_ok=True)
os.makedirs(os.path.dirname(args.outputs_file), exist_ok=True)
logfile = os.path.dirname(args.outputs_file) + f"/{args.jobid}.log"
logging.basicConfig(filename=logfile, level=logging.DEBUG) #, encoding='utf-8'
class DataCollatorForNLI:  

    def __init__(self, tokenized_dataset, device) -> None: # what is label_features? I assume this is the examples
        self.tokenized_dataset = tokenized_dataset
        self.device = device


    def __call__(self, features): # tokenized inputs ->  
        labels = torch.Tensor([feature['labels'] for feature in features]) # get all labels from features
        new_features = {'input_ids': [], 'attention_mask': [], 'label_mask': [], 'cf_input_ids': [], 'cf_attention_mask': [], 'cf_label_mask': []} # we need three things 

        # print("^^^^^^^^^^^^^^^^")
        # print(features)
        for feature in features: # features is (n, num_examples, num_class, total_perm_sets, max_len)?
            new_features['input_ids'].append(feature['input_ids'])
            new_features['attention_mask'].append(feature['attention_mask'])
            new_features['label_mask'].append(feature['label_mask'])
            # new_features['cf_input_ids'].append(feature['cf_input_ids'])
            # new_features['cf_attention_mask'].append(feature['cf_attention_mask'])
            # new_features['cf_label_mask'].append(feature['cf_label_mask'])

        
        tmp = torch.stack(new_features['input_ids'], dim=0)
        # cf_tmp = torch.stack(new_features['cf_input_ids'], dim=0)
        n, num_examples, num_class, total_perm_sets, length = tmp.shape
        # _, _, _, _, c, cf_length = cf_tmp.shape
        
        batch = { # make the dictionary
                    'input_ids': tmp.view(n, num_examples, num_class, total_perm_sets, length).to(self.device), 
                    'attention_mask': torch.stack(new_features['attention_mask'], dim=0).view(n, num_examples, num_class, total_perm_sets, length).to(self.device),
                    'labels': labels.to(self.device),
                    'label_mask': torch.stack(new_features['label_mask'], dim=0).view(n, num_examples, num_class, total_perm_sets, length).to(self.device),
                    # 'cf_input_ids': cf_tmp.view(nw, num_examples, num_class, total_perm_sets, c, cf_length).to(self.device), 
                    # 'cf_attention_mask': torch.stack(new_features['cf_attention_mask'], dim=0).view(n, num_examples, num_class, total_perm_sets, c, cf_length).to(self.device),
                    # 'cf_label_mask': torch.stack(new_features['cf_label_mask'], dim=0).view(n, num_examples, num_class, total_perm_sets, c, cf_length).to(self.device)
                }

        return batch


def get_nll(model, tokenizer, batch, label_mask, num_labels, num_labelstrings, _lambda=10): # lower means more likely, so here get_nll would be the slope of the last '_lambda' layers
    def nll_calc(logits):
        shift_logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1).contiguous()

        shift_target = batch['input_ids'][..., 1:].contiguous()
        # print(shift_target.shape)
        nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none").view(-1, shift_target.size(-1)) # to ensure they correspond and pick the correct word from the dist. reduction is to not do anything extra
        # reshape and undo reshape
        # print("##################")
        # print(nll.shape)
        # print(label_mask.shape)
        # print(nll)
        nll = nll * label_mask[..., 1:]
        # print(nll)
        # input()
        # deno = (label_mask[..., 1:]).sum(dim=-1)
        # deno = (shift_target.ne(tokenizer.pad_token_id).float()).sum(dim=-1) # just to account for the lengths, not too important as lengths of x are the same

        # print(deno)
        # input(deno)
        nll = nll.sum(dim=-1)#/deno 
        # logging.info(nll)
        # input()
        nll = nll.view(-1, num_labels, num_labelstrings)
        return nll

    assert (_lambda >= 2)
    outputs = model(**batch, output_hidden_states=True)
    layers = torch.Tensor([nll_calc(model.lm_head(hidden_state)).item() for hidden_state in outputs.hidden_states])[-_lambda:] # num_layers, num_labels
    return layers

def main():
    # np.random.seed(2024)
    print("OK", args.overwrite)
    print(TOKEN)
    try:
        with open(args.results_file) as fresults_exist:
            # print(args.results_file)
            # print(len(fresults_exist.readlines()), args.num_sets)
            # x = len(fresults_exist.readlines())
            # y = args.num_sets
            # print(x >= y)
            # print(len(fresults_exist.readlines()) >= args.num_sets)
            if len(fresults_exist.readlines()) >= args.num_sets and not args.overwrite:
                print("YES")
                print(f'{args.results_file} already exists and is full. exiting.')
                logging.info(f'{args.results_file} already exists and is full. exiting.')
                return
    except Exception as e:
        pass 
    # input()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    task_items = args.task.split("#")

    if "allenai/OLMo-7B" == args.model:
        tokenizer = hf_olmo.OLMoTokenizerFast.from_pretrained("allenai/OLMo-7B")
    else: 
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir="models/", token=TOKEN)
    print([tokenizer(l) for l in args.label_names.split(",")])
    logging.info(tokenizer.eos_token)
    logging.info(tokenizer.bos_token)
    logging.info(tokenizer.pad_token)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token 
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("OK0")
    
    print("OK1")
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
        print("else")
        print(args.model)
        if not args.device_map:
            print("Loading model")
            if "allenai/OLMo-7B" == args.model:
                model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", token=TOKEN, revision="main")
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", token=TOKEN)#, device_map="auto")
            # print("passed")
        else:
            if "allenai/OLMo-7B" == args.model:
                model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", device_map="auto", token=TOKEN, revision="main")
            else:
                model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir="models/", device_map="auto", token=TOKEN)
        # print("U")
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
    model.config.output_hidden_states = True
    logging.info(task_items[1:])
    suffix = "-text" if args.text else ""
    
    total_perm_sets = args.num_sets 

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
        if args.dataset != "lukaemon/mmlu":
            if args.cat:
                raw_dataset = create_cat(args.dataset, args.data_dir, split=args.split, text1=args.textfield1, text2=args.textfield2, labels=args.labelfield, seed=args.cat_seed)
            else:
                raw_dataset = load_dataset(args.dataset, args.data_dir, split=args.split, data_files=data_files, cache_dir="datasets")
            test_cap = 1000
            if len(raw_dataset) > test_cap:
                raw_dataset = Dataset.from_dict(raw_dataset.shuffle(seed=HF_SHUFFLE_SEED)[:test_cap])
        # else:
        #     raw_dataset = get_mmlu("test")
    if args.dataset != "lukaemon/mmlu":
        raw_train_dataset = load_dataset(args.dataset, args.data_dir, split="train", data_files=None, cache_dir="datasets")
        if len(raw_dataset) > 1000:
            raw_train_dataset = Dataset.from_dict(raw_train_dataset.shuffle(seed=HF_SHUFFLE_SEED)[:1000])
    else: 
        raw_train_dataset= get_mmlu("train")
    # tokenized_dataset = get_tokenized_dataset(raw_dataset, "sentence", "label")
    print("dataseted")
    # print("***************")
    # for t1, t2 in zip(raw_dataset[args.textfield1][:10], raw_dataset[args.textfield2][:10]):
    #     print(t1, t2)
    #     print()
    # input()
    cf_strs = ["N/A", "", "[MASK]"]
    num_cf = len(cf_strs)
    label2id = None
    if args.label2id is not None:
        label2id = eval(args.label2id)
    create = True
    preprocess_path = f"saved_preprocessed/{args.dataset}-{args.data_dir}"
    if len(args.label_names.split(",")) == 2:
        print("YES")
        args.k = max((args.k * 3) // 2, 1)

    custom = ("hGivenP" in args.task or "pGivenH" in args.task)
    print("custom:", custom, args.k)
    # if not args.cat:
    if args.type_of_task == "nli_fewshot":
        if custom:
            tokenized_dataset = get_tokenized_dataset_nli_fewshot_custom(raw_dataset, raw_train_dataset, tokenizer, args.textfield1, args.textfield2, args.labelfield, label2id, args.task, args.k, args.num_sets, args.label_names, create, preprocess_path, FEWSHOT_SEED)
        else:
            tokenized_dataset = get_tokenized_dataset_nli_fewshot(raw_dataset, raw_train_dataset, tokenizer, args.textfield1, args.textfield2, args.labelfield, label2id, args.task, args.k, args.num_sets, args.label_names, create, preprocess_path, FEWSHOT_SEED)
    elif args.type_of_task == "mcq_fewshot":
        tokenized_dataset = get_tokenized_dataset_mcq_fewshot(raw_dataset, raw_train_dataset, tokenizer, args.question, args.choices, args.labelfield, label2id, args.task, args.k, args.num_sets, args.possible_labels, create, preprocess_path, args.want_choice, FEWSHOT_SEED)
    elif args.type_of_task == "mcq_diff_fewshot":
        tokenized_dataset = get_tokenized_dataset_mcq_diff_fewshot(raw_dataset, raw_train_dataset, tokenizer, args.question, args.choices, args.labelfield, label2id, args.task, args.k, args.num_sets, create, preprocess_path, args.want_choice, FEWSHOT_SEED)
    elif args.type_of_task == "mcq_context_fewshot":
        tokenized_dataset = get_tokenized_dataset_mcq_context_fewshot(raw_dataset, raw_train_dataset, tokenizer, args.question, args.passage, args.labelfield, label2id, args.task, args.k, args.num_sets, args.possible_labels, create, preprocess_path, FEWSHOT_SEED)
    
    tok_x = tokenizer.encode("x", add_special_tokens=False)
    tok_labels = [tokenizer.encode("x " + label_name, add_special_tokens=False)[len(tok_x):] for label_name in args.label_names.split(",")]
    
    # calculate all the cc's here
    if not custom:
        all_cfs = {
            "cf_input_ids": tokenized_dataset["cf_input_ids"][0].to(device),
            "cf_attention_mask": tokenized_dataset["cf_attention_mask"][0].to(device),
            "cf_label_mask": tokenized_dataset["cf_label_mask"][0].to(device)
        }
        cf_shape = all_cfs["cf_input_ids"].shape

        all_cc = torch.zeros((args._lambda - 1, cf_shape[1], cf_shape[2])).to(device)

        # for i in range(cf_shape[0]): # k
        for j in range(cf_shape[1]): # num_classes
            for k in range(cf_shape[2]): # num_sets
                tmp_cfs = torch.zeros(cf_shape[3], args._lambda)
                for l in range(cf_shape[3]): # num cf_strings
                    cf_batch = {}
                    cf_batch["input_ids"] = all_cfs["cf_input_ids"][args.k, j, k, l, :].unsqueeze(0)
                    cf_batch["attention_mask"] = all_cfs["cf_attention_mask"][args.k, j, k, l, :].unsqueeze(0)
                    cf_label_mask = all_cfs["cf_label_mask"][args.k, j, k, l, :].unsqueeze(0)
                    tmp_cfs[l, :] = get_nll(model, tokenizer, cf_batch, cf_label_mask, 1, 1, args._lambda)
                # print(tmp_cfs)

                cf_avg = (torch.logsumexp(-torch.Tensor(tmp_cfs), dim=0))
                # print(cf_avg.shape, torch.log(torch.Tensor(num_cf)), num_cf)
                cf_avg = np.log(num_cf) - cf_avg 
                # print(cf_avg.shape, all_cc.shape)
                # input()
                all_cc[:, j, k] = cf_avg
        # all_cc = all_cc.to(device)

    num_labels = tokenized_dataset['input_ids'].shape[2]
    if args.effective_batch_size is not None:
        args.batch_size = max(1, args.effective_batch_size // (total_perm_sets * num_labels))
        print(f"effective batch size: {args.effective_batch_size}, total labelstrings: {total_perm_sets * num_labels}, batch size: {args.batch_size}") # account for the paraphrases
    
    np.random.seed(991)
    logging.info("datasets and tokenizer loaded")

    data_collator = DataCollatorForNLI(tokenized_dataset, device) # don't need to pass in tokenized_dataset
    # data_collator.tokenizer.pad_token_id = tokenizer.eos_token_id
    eval_dataloader = DataLoader(tokenized_dataset, collate_fn=data_collator, batch_size=args.batch_size) # it can iterate on its own

    logging.info("starting evaluation now....")

    fresults = open(args.results_file, "w")
    foutputs = open(args.outputs_file, "w")

    accurate = { 
        'results': [[0 for _ in range(args._lambda)] for setid in range(args.num_sets)],
        }

    all_predictions = {
        'results': [[[] for _ in range(args._lambda)] for setid in range(args.num_sets)],
        }

    cc_accurate = { 
        'results': [[0 for _ in range(args._lambda)] for setid in range(args.num_sets)],
        }

    cc_all_predictions = {
        'results': [[[] for _ in range(args._lambda)] for setid in range(args.num_sets)],
        }


    total = 0
    all_labels = [] 

    from pprint import pprint 
    init_index = 1
    if args.debug:
        init_index = total_perm_sets
    end_index = total_perm_sets+1
    results = []
    nlls_ent = []
    nlls_not_ent = []
    with torch.no_grad():            
        # for num_examples in range(args.k+1):
        num_examples = args.k
        for batch in tqdm(eval_dataloader):
            
            subbatches = []
            label_masks = []
            labels = batch['labels'].to(device)
            del batch['labels']

            label_mask = batch['label_mask']
            del batch['label_mask']
            # if num_examples == 0:
            all_labels += labels.tolist()
            total += labels.size(0)
            # print("batch shape", batch['input_ids'].shape)
            

            if args.batch_by_labelstring:
                nll = torch.empty(batch['input_ids'].shape[0], num_labels, total_perm_sets, args._lambda).to(device)
                if not custom:
                    cc_nll = torch.empty(batch['input_ids'].shape[0], num_labels, total_perm_sets, args._lambda).to(device)
                for i in range(num_labels): # for each label
                    for j in range(total_perm_sets): # for each prompt per label
                        sub_batch={}
                        max_len = torch.max(batch['attention_mask'][:, num_examples, i, j, :].sum(dim=-1))
                        sub_batch['input_ids'], sub_batch['attention_mask'], label_mask_ij = batch['input_ids'][:, num_examples, i, j, :max_len], batch['attention_mask'][:, num_examples, i, j, :max_len], label_mask[:, num_examples, i, j, :max_len]
                        test_prd = label_mask_ij * sub_batch['input_ids']
                        val = get_nll(model, tokenizer, sub_batch, label_mask_ij, 1, 1, args._lambda)
                        nll[:, i, j, :] = val.squeeze()
                        if not custom:
                            cc_nll[:, i, j, :] = nll[:, i, j, :] - all_cc[:, i, j]

            else:
                nll = get_nll(model, tokenizer, batch, label_mask, num_labels, total_perm_sets, args._lambda)

            
            if args.debug:
                new_batch_text = tokenizer.batch_decode(batch['input_ids'])
                for i in range(nll.size(2)):
                    r = num_labels * total_perm_sets
                    input(f"debugging {i}")
            for num_layers_in_calc in range(args._lambda - 1):
                for setid in range(args.num_sets):  # we only compute probs once and compute mean and var by grouping them for the x-axis values              
                    # ids = torch.from_numpy(np.random.choice(np.arange(total_perm_sets), num_examples+1, replace=False)).to(device)
                    # print(nll.shape)
                    # print(ids)
                    nll_subset = nll[:, :, setid, num_layers_in_calc]
                    
                    if args.debug:
                        print(labels)
                    # loss = -torch.logsumexp(-nll_subset, dim=2) + np.log(total_perm_sets) # summing over nl probabilities. To sum, need to convert nll to ll, then exponentiate, them sum, then log again. To prevent underflow (batch size; no. labels).T
                    results = nll_subset.min(dim=1)[1] # an array of label indices, i.e. the prediction
                    if not custom:
                        cc_nll_subset = cc_nll[:, :, setid, num_layers_in_calc]
                        cc_results = cc_nll_subset.min(dim=1)[1]

                    # print("###", result_logsumexp.shape)
                    # input()
                    # if args.debug:
                    #     print(loss, result_logsumexp)

                    #average or geometric mean of probabilities
                    # loss = torch.mean(nll_subset, dim=2) # just different processing
                    # result_average = loss.min(dim=1)[1]
                    # if args.debug:
                    #     print(loss, result_average)

                    # #harmonic mean of probabilities
                    # loss = -np.log(total_perm_sets) + torch.logsumexp(nll_subset, dim=2)
                    # result_vote = loss.min(dim=1)[1]

                    #vote
                    # result_vote = nll_subset.min(dim=0)[1].mode(dim=0)[0]
                    # logging.info(nll_subset.min(dim=0)[1])
                    # if args.debug:   
                    #     logging.info(loss, result_vote)
                    #     input()

                    accurate['results'][setid][num_layers_in_calc] += results.eq(labels).int().sum().item()
                    
                    # accurate['average'][setid][num_examples] += result_average.eq(labels).int().sum().item()
                    # accurate['vote'][setid][num_examples] += result_vote.eq(labels).int().sum().item()

                    all_predictions['results'][setid][num_layers_in_calc] += results.tolist()
                    
                    if not custom:
                        cc_accurate['results'][setid][num_layers_in_calc] += cc_results.eq(labels).int().sum().item()
                        cc_all_predictions['results'][setid][num_layers_in_calc] += cc_results.tolist()
                    # all_predictions['average'][setid][num_examples] += result_average.tolist()
                    # all_predictions['vote'][setid][num_examples] += result_vote.tolist()
                    # print(all_predictions['logsumexp'])
                    # print(result_logsumexp)
                    # print()
                    # input()
    # for setid in range(args.num_sets):
    #     for num_layers_in_calc in range(args._lambda - 1):
    #         print(len(all_predictions['results'][setid][num_layers_in_calc]), setid, num_layers_in_calc)
    # input()
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
        
    for setid in range(args.num_sets):
        result = { 
            "k": [],
            'f1': [],
            'accuracy': [],
            'confusion_matrix': [],
            'cc_f1': [],
            'cc_accuracy': [],
            'cc_confusion_matrix': []
        }
        for layer_num in range(args._lambda):

            # print(len(all_labels), len(all_predictions['results'][setid][layer_num]))
            cm = confusion_matrix(all_labels, all_predictions['results'][setid][layer_num])
            if not custom:
                cc_cm = confusion_matrix(all_labels, cc_all_predictions['results'][setid][layer_num])

            result["k"].append(layer_num)
            for metric in ('f1', 'accuracy'):
                result[metric].append(compute_metric(cm, metric))
                if not custom:
                    result['cc_'+metric].append(compute_metric(cc_cm, metric))
            result['confusion_matrix'].append(str(cm))
            if not custom:
                result['cc_confusion_matrix'].append(str(cc_cm))
            
            logging.info(f"confusion matrix: \n{cm}")
        fresults.write(json.dumps(result) + "\n")
        
        outputfile = os.path.dirname(args.outputs_file) + f"/run-{setid}_" + os.path.basename(args.outputs_file)
        logging.info(outputfile)
        with open(outputfile, "w") as foutputs:
            predictions = [" ".join(map(str, item)) for item in zip(*all_predictions['results'][setid])]
            outputs = [f"{label} {output}" for label, output in zip(all_labels, predictions)]
            foutputs.write("\n".join(outputs) + "\n")
        
        if not custom:
            cc_outputfile = os.path.dirname(args.cc_outputs_file) + f"/run-{setid}_" + os.path.basename(args.cc_outputs_file)
            logging.info(cc_outputfile)
            with open(cc_outputfile, "w") as foutputs:
                cc_predictions = [" ".join(map(str, item)) for item in zip(*cc_all_predictions['results'][setid])]
                cc_outputs = [f"{label} {output}" for label, output in zip(all_labels, cc_predictions)]
                foutputs.write("\n".join(cc_outputs) + "\n")


if __name__=="__main__":
    main()  
