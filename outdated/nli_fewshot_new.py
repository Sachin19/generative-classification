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
from config_fewshot_new import TASK2LABELSTRINGS, EXAMPLEFORMAT2ENTAIL, EXAMPLEFORMAT2NOTENTAIL, EXAMPLEFORMAT_SPACE2ENTAIL, EXAMPLEFORMAT_SPACE2NOTENTAIL#, EXAMPLEFORMAT2, EXAMPLEFORMAT2_SPACE

from dataset_loaders import TASK2LOADER, TOKEN

import logging

def get_tokenized_dataset_nli_fewshot(raw_dataset, raw_train_dataset, tokenizer, textfield1="sentence1", textfield2="sentence2", labelfield="label", label2id=None, task="", _k=10, _num_sets=8, _label_names="", create=False, preprocess_path="", few_shot_seed=2024, cfs=['N/A', '', '[MASK]']):
    def preprocess_function(examples):  
        np.random.seed(few_shot_seed)          
        print("creating")
        print("****************")
        pad_token_id = tokenizer.pad_token_id
        template = TASK2LABELSTRINGS[task][0]
        label_names = _label_names.split(",")
        num_classes = len(label_names)
        num_cf = len(cfs)

        premises = np.array(examples[textfield1])
        hypothesis = np.array(examples[textfield2])
        labels = np.array(examples[labelfield])

        n = len(labels)

        # get few shot examples
        few_shot_examples = np.empty((_k+1, _num_sets), dtype=object)
        train_data = raw_train_dataset
        train_premises = np.array(train_data[textfield1])
        train_hypothesis = np.array(train_data[textfield2])
        train_labels = np.array(train_data[labelfield])
        
        for i in range(_num_sets):
            print(f"generating fewshot examples {i}/{_num_sets}", end="\r")
            idxs = np.random.choice(range(0, len(train_premises)), _k, replace=False)
            few_shot = {}
            few_shot["premise"] = train_premises[idxs]
            few_shot["hypothesis"] = train_hypothesis[idxs]
            few_shot["label"] = train_labels[idxs]
            few_shot_strs = np.empty((_k), dtype=object)
            # print(len(few_shot["label"]))
            # print(len(few_shot["premise"]))
            # print(len(few_shot["hypothesis"]))
            # print(_k)
            
            # print("Here!")
            # input()
            for j in range(_k):
                # print(label_names[few_shot["label"][j]])
                few_shot_strs[j] = template.format(premise=few_shot["premise"][j], hypothesis=few_shot["hypothesis"][j], label=label_names[few_shot["label"][j]])
            
            for num_examples in range(_k+1):
                example_idxs = np.random.choice(range(0, _k), num_examples, replace=False)
                few_shot_strs_num_examples = few_shot_strs[example_idxs]

                tmp_str = ""
                    
                for k in range(num_examples):
                    tmp_str += few_shot_strs_num_examples[k] + "\n\n"
                few_shot_examples[num_examples, i] = tmp_str
        
        total_examples = _num_sets
        tokens = np.full((n, _k+1, num_classes, total_examples), None)
        cf_tokens = np.full((n, _k+1, num_classes, total_examples, num_cf), None)
        attention_masks = np.full((n, _k+1, num_classes, total_examples), None)
        cf_attention_masks = np.full((n, _k+1, num_classes, total_examples, num_cf), None)
        label_masks = np.full((n, _k+1, num_classes, total_examples), None)
        cf_label_masks = np.full((n, _k+1, num_classes, total_examples, num_cf), None)
        tok_x = tokenizer.encode("x", add_special_tokens=False)
        for i in range(n):
            # print(f"tokenizing {i}/{n}", end="\r")
            quoted_premise = premises[i]
            quoted_hypothesis = hypothesis[i]
            for num_examples in range(_k+1):
                for j in range(num_classes):
                    label = label_names[j]
                     
                    tokenized_label = tokenizer.encode("x " + label, add_special_tokens=False)[len(tok_x):]
                    for k in range(total_examples):
                        string = few_shot_examples[num_examples, k] + template.format(premise=quoted_premise, hypothesis=quoted_hypothesis, label=label)
                        
                        tmp_tok = tokenizer(string)
                        tokens[i, num_examples, j, k] = np.array(tmp_tok['input_ids'])
                        attention_masks[i, num_examples, j, k] = np.array(tmp_tok['attention_mask'])
                        tmp_label_mask = np.zeros_like(tokens[i, num_examples, j, k])
                        
                        idx = len(tokenized_label)
                        tmp_label_mask[-idx:] = 1
                        label_masks[i, num_examples, j, k] = tmp_label_mask

                        ## do the cf string now
                        for c in range(num_cf):
                            cf_string = few_shot_examples[num_examples, k] + template.format(premise=cfs[c], hypothesis=cfs[c], label=label)

                            cf_tmp_tok = tokenizer(cf_string)
                            cf_tokens[i, num_examples, j, k, c] = np.array(cf_tmp_tok['input_ids'])
                            cf_attention_masks[i, num_examples, j, k, c] = np.array(cf_tmp_tok['attention_mask'])
                            cf_tmp_label_mask = np.zeros_like(cf_tokens[i, num_examples, j, k, c])

                            idx = len(tokenized_label)
                            cf_tmp_label_mask[-idx:] = 1
                            cf_label_masks[i, num_examples, j, k, c] = cf_tmp_label_mask
                            # print(cf_string)
                            # input()

                        # if i == 5 and num_examples in (1, 7) and k in (1, 5):
                        # print(i, num_examples, j, k)
                        # print(tokens[i, num_examples, j, k])
                        # print(attention_masks[i, num_examples, j, k])
                        # print(label_masks[i, num_examples, j, k])
                        # print(label)
                        # print(tokenized_label)
                        # print(idx)
                        # tmp = np.array(label_masks[i, num_examples, j, k]) * np.array(tokens[i, num_examples, j, k])
                        # print(tokenizer.decode(tmp[tmp != 0]))

                        # print(label)
                        # print(i, num_examples, j, k)
                        # print(string)
                        # print()
                        # input()
        max_len = 0 # it's safe to assume that the max_len for the actual strings will be greater than the max_len for the cf, as we are replacing the hypothesis and premise by one word
        
        for i in range(n):
            for num_examples in range(_k+1):
                for j in range(num_classes):
                    for k in range(total_examples):
                        curr_length = len(tokens[i, num_examples, j, k])
                        if curr_length > max_len:
                            max_len = curr_length
        
        padded_tokens = np.full((n, _k+1, num_classes, total_examples, max_len), pad_token_id)
        padded_attention_mask = np.full((n, _k+1, num_classes, total_examples, max_len), 0)
        padded_label_mask = np.full((n, _k+1, num_classes, total_examples, max_len), 0)

        cf_padded_tokens = np.full((n, _k+1, num_classes, total_examples, num_cf, max_len), pad_token_id)
        cf_padded_attention_mask = np.full((n, _k+1, num_classes, total_examples, num_cf, max_len), 0)
        cf_padded_label_mask = np.full((n, _k+1, num_classes, total_examples, num_cf, max_len), 0)
                
        for i in range(n):
            # print(f"padding {i}/{n}", end="\r")
            for num_examples in range(_k+1):
                for j in range(num_classes):
                    for k in range(total_examples):
                        padded_tokens[i, num_examples, j, k, :len(tokens[i, num_examples, j, k])] = tokens[i, num_examples, j, k]
                        padded_attention_mask[i, num_examples, j, k, :len(attention_masks[i, num_examples, j, k])] = attention_masks[i, num_examples, j, k]
                        padded_label_mask[i, num_examples, j, k, :len(label_masks[i, num_examples, j, k])] = label_masks[i, num_examples, j, k]

                        for c in range(num_cf):
                            cf_padded_tokens[i, num_examples, j, k, c, :len(cf_tokens[i, num_examples, j, k, c])] = cf_tokens[i, num_examples, j, k, c]
                            cf_padded_attention_mask[i, num_examples, j, k, c, :len(cf_attention_masks[i, num_examples, j, k, c])] = cf_attention_masks[i, num_examples, j, k, c]
                            cf_padded_label_mask[i, num_examples, j, k, c, :len(cf_label_masks[i, num_examples, j, k, c])] = cf_label_masks[i, num_examples, j, k, c]
        # print("%%%%%%%")
        # print(labels[:10])
        # print(padded_tokens[:10, 0, 0, :])
        tokenized_dataset = {
            'input_ids': torch.from_numpy(padded_tokens), # (n, num_classes, total_examples, maxlen)
            'attention_mask': torch.from_numpy(padded_attention_mask), # (n, num_classes, total_examples, maxlen)
            'label_mask': torch.from_numpy(padded_label_mask), # (n, num_classes, total_examples, maxlen)
            'labels': torch.from_numpy(labels), # (n)
            'cf_input_ids': torch.from_numpy(cf_padded_tokens), # (n, num_classes, total_examples, c, maxlen)
            'cf_attention_mask': torch.from_numpy(cf_padded_attention_mask), # (n, num_classes, total_examples, c, maxlen)
            'cf_label_mask': torch.from_numpy(cf_padded_label_mask), # (n, num_classes, total_examples, c, maxlen)
        }
        for key, val in tokenized_dataset.items():
            print(key, ": ", val.shape)
        # print("***************")
        # for key, val in tokenized_dataset.items():
        #     print(key, ":", val.shape)
        # input()
        # print(tokenized_dataset['input_ids'].shape, tokenized_dataset['labels'].shape)
        # test_i = 2
        # test_j = 0
        # test_examples = 3
        # test_k = 2
        # # s_tmp = few_shot_examples[test_examples - 1, k] + template.format(premise=quoted_premise, hypothesis=quoted_hypothesis, label=label_names[test_j])
        # quoted_premise = "\"" + premises[test_i] + "\""
        # quoted_hypothesis = "\"" + hypothesis[test_i] + "\""
        # print(tokenizer(s_tmp))
        # print(len(np.where(padded_tokens[test_i, test_j, test_k, :] != pad_token_id)[0]))
        # print("^^^^^^^")
        # print(template.format(premise=quoted_premise, hypothesis=quoted_hypothesis, label=label_names[test_j]))
        # print(np.where(padded_tokens[test_i, test_j, test_k, :]==tokenizer(label_names[test_j])))
        # print(padded_tokens[test_i, test_examples, test_j, test_k, :])
        # print(tokenizer('True', add_special_tokens=False))
        # print(tokenizer('False', add_special_tokens=False))
        # print(padded_label_mask[test_i, test_examples, test_j, test_k, :])
        # print(np.where(padded_label_mask[test_i, test_j, test_k, :] == 1))
        # print(padded_attention_mask[test_i, test_examples, test_j, test_k, :])
        # val = np.dot(padded_tokens[test_i, test_examples, test_j, test_k, :], padded_label_mask[test_i, test_examples, test_j, test_k, :])
        # print("****", val)
        # print("\"" + str(tokenizer.decode(val)) + "\"")
        # print(tokenizer.decode([25]))
        # print(s_tmp.format(text1=quoted_masked, text2=quoted_target))
        # # print(splitted[0][3])
        # print(quoted_target)
        # print(tokenizer(quoted_target)["input_ids"])
        # print(padded_tokenized_targets[test_i, test_j, test_k])
        # print(quoted_masked)
        # print(tokenizer(quoted_masked, add_special_tokens=False)["input_ids"])

        return tokenized_dataset

    
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    
    columns_to_remove = raw_dataset.column_names
    if label2id is None:
        columns_to_remove.remove(labelfield)
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    logging.info(tokenized_dataset)
    tokenized_dataset.set_format("torch")

    return tokenized_dataset