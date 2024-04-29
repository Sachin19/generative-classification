# this differs from arc.py as arc.py has all answers in one column, while this has it split across multiple
import os
from dataclasses import dataclass, field
from typing import Optional, Literal
from tqdm import tqdm
import json
import re
import random 
import numpy as np
from sklearn.metrics import confusion_matrix
import pickle

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser, BitsAndBytesConfig
from datasets import load_dataset, Dataset, load_from_disk

# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

from config_channel_ablate import TASK2LABELSTRINGS as TASK2ABLATELABELSTRINGS
from config_fewshot_new import TASK2LABELSTRINGS, EXAMPLEFORMAT2ENTAIL, EXAMPLEFORMAT2NOTENTAIL, EXAMPLEFORMAT_SPACE2ENTAIL, EXAMPLEFORMAT_SPACE2NOTENTAIL#, EXAMPLEFORMAT2, EXAMPLEFORMAT2_SPACE

from dataset_loaders import TASK2LOADER, TOKEN

import logging

def get_tokenized_dataset_mmlu(raw_dataset, raw_train_dataset, tokenizer,  _ks=[0, 1, 2, 4, 8], _num_sets=4, create=False, few_shot_seed=2024, cfs=['N/A', '', '[MASK]']):
    choices = "choices"
    labelfield = "target"
    question = "input"
    _task = "mmlu_fewshot"
    label_names = ["A", "B", "C", "D"]
    def preprocess_function(examples): # would need to keep the prem and hyp separate, or just get the mask here
        
        print("creating mmlu")
        np.random.seed(few_shot_seed)   
        examples_choices = np.array(examples[choices]) # n by 4
        examples_questions = examples[question]
        labels = np.array([label_names.index(examples[labelfield][i]) for i in range(len(examples[labelfield]))])
        tok_x = tokenizer.encode("x", add_special_tokens=False)
        
        tok_labels = [tokenizer.encode("x " + label_name, add_special_tokens=False)[len(tok_x):] for label_name in label_names]
        # print(tok_labels)
        pad_token_id = tokenizer.pad_token_id
        template = TASK2LABELSTRINGS[_task][0]
        choice_template = TASK2LABELSTRINGS[_task][1]
        n = len(examples_questions)
        num_classes = 4
        num_cf = len(cfs)
        
        def compile_choices(choices):
            n = len(choices)
            choices_compiled = np.full((n, num_classes), None)
            for i, choice in enumerate(choices):
                choices_compiled[i, :] = choice
            return choices_compiled
        
        # prepare fewshot examples
        few_shot_examples = np.empty((len(_ks), _num_sets), dtype=object)
        
        train_data = raw_train_dataset #load_dataset(args.dataset, args.data_dir, split="train", data_files=None, cache_dir="datasets")
        
        train_questions = np.array(train_data[question])
        # train_choices, train_labels_maxed, num_train_classes = fill_choices(train_data[choices])
        train_choices = np.array(train_data[choices])
        
        # print(train_choices.shape, train_questions.shape)
        # input()
        train_labels = np.array([label_names.index(train_data[labelfield][i]) for i in range(len(train_data[labelfield]))])
        # print(train_labels[:10])
        # input()
        
        for i in range(_num_sets):
            # index into and get the k strings
            idxs = np.random.choice(range(0, len(train_questions)), max(_ks), replace=False)
            few_shot = {}
            few_shot["question"] = train_questions[idxs]
            few_shot["choices"] = train_choices[idxs] # array of length 
            few_shot["label"] = train_labels[idxs]
            
            few_shot_strs = np.full((max(_ks)), "", dtype=object)
            # format all k strings 
            for j, k in enumerate(_ks):
                tmp_choices = ""
                # print(few_shot["choices"][j])
                # input()
                for l in range(4):
                    chs, lb_n =  few_shot["choices"][j][l], label_names[l]
                    # print(chs, lb_n)
                    tmp_choices += choice_template.format(label=lb_n, choice=chs)
                # print("past", choice_template)

                few_shot_strs[j] = template.format(question=few_shot["question"][j], choices=tmp_choices, label=label_names[few_shot["label"][j]])
            
            # compile all the strings from 1 to k and concant into one string (so total k strings)
            for idx, num_examples in enumerate(_ks):
                example_idxs = np.random.choice(range(0, num_examples), num_examples, replace=False)
                # print(example_idxs)
                # input()
                
                tmp_str = ""    
                if num_examples != 0:
                    few_shot_strs_num_examples = few_shot_strs[example_idxs]
                    for k in range(num_examples):
                        tmp_str += few_shot_strs_num_examples[k] + "\n\n"
                few_shot_examples[idx, i] = tmp_str
        # print(few_shot_examples[idx, i])
        # test_choices_maxed, test_labels_maxed, num_classes, num_classes = fill_choices(examples_choices)
        
        # for i in range(len(_ks)):
        #     for j in range(_num_sets):
        #         print(few_shot_examples[i, j])

        #         input()
        total_examples = _num_sets
        tokens = np.full((n, len(_ks), num_classes, total_examples), None)
        attention_masks = np.full((n, len(_ks), num_classes, total_examples), None)
        label_masks = np.full((n, len(_ks), num_classes, total_examples), None)
        cf_tokens = np.full((n, len(_ks), num_classes, total_examples, num_cf), None)
        cf_attention_masks = np.full((n, len(_ks), num_classes, total_examples, num_cf), None)
        cf_label_masks = np.full((n, len(_ks), num_classes, total_examples, num_cf), None)

        # labels = np.full(n, 0)

        for i in range(n):
            # print(f"tokenizing -> {i}/{n}", end="\r")
            
            quoted_question = examples_questions[i] # no more quotes
            tmp_choices = ""
            # labels[i] = np.where(test_labels_maxed[i] == examples[labelfield][i])[0]
            for j in range(num_classes):
                tmp_choices += choice_template.format(label=label_names[j], choice=examples_choices[i, j])

            all_choices = tmp_choices #+ extra_choices
            for num_examples in range(len(_ks)):
                for j in range(num_classes):
                    label = label_names[j]
                    
                    for k in range(total_examples):
                        # if j >= num_classes[i]:
                        #     if want_choice:
                                # label =  np.random.choice(test_choices_maxed[i, :num_classes[i]])
                        #     else: 
                        # label = np.random.choice(examples_choices[i, :num_classes[i]])

                        tokenized_label = tokenizer.encode("x " + label, add_special_tokens=False)[len(tok_x):]
                        string = few_shot_examples[num_examples, k] + template.format(question=quoted_question, choices=all_choices, label=label)
                        
                        tmp_tok = tokenizer(string)
                        tokens[i, num_examples, j, k] = np.array(tmp_tok['input_ids'])
                        attention_masks[i, num_examples, j, k] = np.array(tmp_tok['attention_mask'])
                        tmp_label_mask = np.zeros_like(tokens[i, num_examples, j, k])
                        
                        idx = len(tokenized_label)
                        tmp_label_mask[-idx:] = 1
                        label_masks[i, num_examples, j, k] = tmp_label_mask

                        for c in range(num_cf):
                            cf_string = few_shot_examples[num_examples, k] + template.format(question=cfs[c], choices=cfs[c], label=label)
                            # print(cf_string)

                            cf_tmp_tok = tokenizer(cf_string)
                            cf_tokens[i, num_examples, j, k, c] = np.array(cf_tmp_tok['input_ids'])
                            cf_attention_masks[i, num_examples, j, k, c] = np.array(cf_tmp_tok['attention_mask'])
                            cf_tmp_label_mask = np.zeros_like(cf_tokens[i, num_examples, j, k, c])

                            cf_tmp_label_mask[-idx:] = 1
                            cf_label_masks[i, num_examples, j, k, c] = cf_tmp_label_mask

                        # print(i, num_examples, j, k)
                        # print(tokens[i, num_examples, j, k])
                        # print(attention_masks[i, num_examples, j, k])
                        # print(label_masks[i, num_examples, j, k])
                        # print(label)
                        # print(tokenized_label)
                        # print(idx)

                        # tmp = np.array(label_masks[i, num_examples, j, k]) * np.array(tokens[i, num_examples, j, k])
                        # print("{" + string + "}")
                        # print(tokenizer.decode(tmp[tmp != 0]))
                        # # print(label)
                        # print(i, num_examples, j, k)
                        # print()
                        # input()

        # print()

        max_len = 0
        for i in range(n):
            for num_examples in range(len(_ks)):
                for j in range(num_classes):
                    for k in range(total_examples):
                        curr_length = len(tokens[i, num_examples, j, k])
                        if curr_length > max_len:
                            max_len = curr_length

        padded_tokens = np.full((n, len(_ks), num_classes, total_examples, max_len), pad_token_id)
        padded_attention_mask = np.full((n, len(_ks), num_classes, total_examples, max_len), 0)
        padded_label_mask = np.full((n, len(_ks), num_classes, total_examples, max_len), 0)
        cf_padded_tokens = np.full((n, len(_ks), num_classes, _num_sets, num_cf, max_len), pad_token_id)
        cf_padded_attention_mask = np.full((n, len(_ks), num_classes, _num_sets, num_cf, max_len), 0)
        cf_padded_label_mask = np.full((n, len(_ks), num_classes, _num_sets, num_cf, max_len), 0)
        for i in range(n):
            # print(f"padding: {i}/{n}", end="\r")
            for num_examples in range(len(_ks)):
                for j in range(num_classes):
                    for k in range(total_examples):
                        padded_tokens[i, num_examples, j, k, :len(tokens[i, num_examples, j, k])] = tokens[i, num_examples, j, k]
                        padded_attention_mask[i, num_examples, j, k, :len(attention_masks[i, num_examples, j, k])] = attention_masks[i, num_examples, j, k]
                        padded_label_mask[i, num_examples, j, k, :len(label_masks[i, num_examples, j, k])] = label_masks[i, num_examples, j, k]
                         # test
                        test_prd = padded_label_mask[i, num_examples, j, k, :] * padded_tokens[i, num_examples, j, k, :]
                        # print(f"({tokenizer.decode(test_prd[test_prd != 0])})", f"({tokenizer.decode(tok_labels[j])})")
                        # print()

                        assert (test_prd[test_prd != 0] == tok_labels[j]).all()
                        # print(tokenizer.decode(test_prd[test_prd != 0]))
                        

                        for c in range(num_cf):
                            cf_padded_tokens[i, num_examples, j, k, c, :len(cf_tokens[i, num_examples, j, k, c])] = cf_tokens[i, num_examples, j, k, c]
                            cf_padded_attention_mask[i, num_examples, j, k, c, :len(cf_attention_masks[i, num_examples, j, k, c])] = cf_attention_masks[i, num_examples, j, k, c]
                            cf_padded_label_mask[i, num_examples, j, k, c, :len(cf_label_masks[i, num_examples, j, k, c])] = cf_label_masks[i, num_examples, j, k, c]
                            
                            test_prd_cf = cf_padded_label_mask[i, num_examples, j, k, c, :] * cf_padded_tokens[i, num_examples, j, k, c, :]

                            assert (test_prd_cf[test_prd_cf != 0] == tok_labels[j]).all()
        # print("***************")
        tokenized_dataset = {
            'input_ids': torch.from_numpy(padded_tokens), # (n, num_classes, total_examples, maxlen)
            'attention_mask': torch.from_numpy(padded_attention_mask), # (n, num_classes, total_examples, maxlen)
            'label_mask': torch.from_numpy(padded_label_mask), # (n, num_classes, total_examples, maxlen)
            'labels': torch.from_numpy(labels), # (n)
            'cf_input_ids': torch.from_numpy(cf_padded_tokens),
            'cf_attention_mask': torch.from_numpy(cf_padded_attention_mask),
            'cf_label_mask': torch.from_numpy(cf_padded_label_mask)
        }

        print("***")
        for key, val in tokenized_dataset.items():
            print(key, ": ", val.shape)

        return tokenized_dataset
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, batch_size=len(raw_dataset[labelfield]))
    columns_to_remove = raw_dataset.column_names

    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

    logging.info(tokenized_dataset)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset
