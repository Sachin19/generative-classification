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

def get_tokenized_dataset_mcq_diff_fewshot(raw_dataset, raw_train_dataset, tokenizer, question="question", choices="choices", labelfield="label", label2id=None,  _task="", _k=10, _num_sets=8, create=False, preprocess_path="", want_choice=True, few_shot_seed=2024):
    def preprocess_function(examples): # would need to keep the prem and hyp separate, or just get the mask here
        # if os.path.isfile(f"{preprocess_path}-input_ids.pt") and not create:
        #     print("loading...")
        #     tokenized_dataset = {"input_ids": None, "attention_mask": None, "label_mask": None, "labels": None}
        #     for key in tokenized_dataset.keys():
        #         tokenized_dataset[key] = torch.load(f'{preprocess_path}-{key}.pt')
        #     return tokenized_dataset

        print("creating")
        np.random.seed(few_shot_seed)   
        examples_choices = examples[choices]
        examples_questions = examples[question]
        
        pad_token_id = tokenizer.pad_token_id
        template = TASK2LABELSTRINGS[_task][0]
        choice_template = TASK2LABELSTRINGS[_task][1]
        n = len(examples_questions)

        def fill_choices(choices):
            # there are differences in the number of answers to each question
            n = len(choices)
            max_num_classes = 0
            num_classes = np.full(n, 0)
            for i in range(n):
                l_for_length_calc = len(choices[i]["text"])
                num_classes[i] = l_for_length_calc
                # if l_for_length_calc < 4:
                #     print("$$$$$$$$$")
                #     print(choices[i])
                if l_for_length_calc > max_num_classes:
                    max_num_classes = l_for_length_calc
                    # print("$$$$$$$$$$")
                    # print(choices[i])

            choices_maxed = np.full((n, max_num_classes), None)
            labels_maxed = np.full((n, max_num_classes), None)
            for i in range(n):
                
                availables = choices[i]["text"]
                labels = choices[i]["label"]
                choices_maxed[i, :len(availables)] = availables
                labels_maxed[i, :len(labels)] = labels
                if len(availables) < max_num_classes:
                    for j in range(len(availables), max_num_classes):
                        choices_maxed[i, j] = np.random.choice(availables)
                        labels_maxed[i, j] = chr(ord(labels_maxed[i, 0])+j)
            return choices_maxed, labels_maxed, max_num_classes, num_classes
        
        def compile_choices(choices):
            n = len(choices)
            choices_compiled = np.full((n), None)
            labels_compiled = np.full((n), None)
            num_classes = 0
            for i, choice in enumerate(choices): 
                if len(choice["text"]) > num_classes:
                    num_classes = len(choice["text"])
                choices_compiled[i] = choice["text"]
                labels_compiled[i] = choice["label"]
            return choices_compiled, labels_compiled, num_classes
        
        
        
        # prepare fewshot examples
        few_shot_examples = np.empty((_k+1, _num_sets), dtype=object)
        
        train_data = raw_train_dataset #load_dataset(args.dataset, args.data_dir, split="train", data_files=None, cache_dir="datasets")
        
        train_questions = np.array(train_data[question])
        # train_choices, train_labels_maxed, num_train_classes = fill_choices(train_data[choices])
        train_choices, train_labels_maxed, _ = compile_choices(train_data[choices])     
        train_labels = np.array([list(train_labels_maxed[i]).index(train_data[labelfield][i]) for i in range(len(train_data))])
        
        for i in range(_num_sets):
            # index into and get the k strings
            idxs = np.random.choice(range(0, len(train_questions)), _k, replace=False)
            few_shot = {}
            few_shot["question"] = train_questions[idxs]
            few_shot["choices"] = train_choices[idxs]
            few_shot["label_names"] = train_labels_maxed[idxs]
            few_shot["label"] = train_labels[idxs]
            
            few_shot_strs = np.full((_k), "", dtype=object)
            # format all k strings 
            for j in range(_k):
                tmp_choices = ""
                for chs, lb_n in zip(few_shot["choices"][j], few_shot["label_names"][j]): 
                    tmp_choices += choice_template.format(label=lb_n, choice=chs)
                if want_choice:
                    lab = few_shot["choices"][j][few_shot["label"][j]]
                else:
                    lab = few_shot["label_names"][j][few_shot["label"][j]]
                few_shot_strs[j] = template.format(question=few_shot["question"][j], choices=tmp_choices, label=lab)
            
            # compile all the strings from 1 to k and concant into one string (so total k strings)
            for num_examples in range(_k+1):
                example_idxs = np.random.choice(range(0, _k), num_examples, replace=False)
                few_shot_strs_num_examples = few_shot_strs[example_idxs]
                tmp_str = ""    
                for k in range(num_examples):
                    tmp_str += few_shot_strs_num_examples[k] + "\n\n"
                few_shot_examples[num_examples, i] = tmp_str

        test_choices_maxed, test_labels_maxed, max_num_classes, num_classes = fill_choices(examples_choices)

        total_examples = _num_sets
        tokens = np.full((n, _k+1, max_num_classes, total_examples), None)
        attention_masks = np.full((n, _k+1, max_num_classes, total_examples), None)
        label_masks = np.full((n, _k+1, max_num_classes, total_examples), None)
        labels = np.full(n, 0)
        tok_x = tokenizer.encode("x", add_special_tokens=False)

        for i in range(n):
            print(f"tokenizing -> {i}/{n}", end="\r")
            
            quoted_question = examples_questions[i] # no more quotes
            tmp_choices = ""
            labels[i] = np.where(test_labels_maxed[i] == examples[labelfield][i])[0]
            for j in range(num_classes[i]):
                tmp_choices += choice_template.format(label=test_labels_maxed[i, j], choice=test_choices_maxed[i, j])

            all_choices = tmp_choices #+ extra_choices
            for num_examples in range(_k+1):
                for j in range(max_num_classes):
                    if j < num_classes[i]:
                        if want_choice:
                            label = test_choices_maxed[i, j]
                        else:
                            label = test_labels_maxed[i, j]
                    
                    for k in range(total_examples):
                        if j >= num_classes[i]:
                            if want_choice:
                                label =  np.random.choice(test_choices_maxed[i, :num_classes[i]])
                            else: 
                                label =  np.random.choice(test_labels_maxed[i, :num_classes[i]])

                        tokenized_label = tokenizer.encode("x " + label, add_special_tokens=False)[len(tok_x):]
                        string = few_shot_examples[num_examples, k] + template.format(question=quoted_question, choices=all_choices, label=label)
                        
                        tmp_tok = tokenizer(string)
                        tokens[i, num_examples, j, k] = np.array(tmp_tok['input_ids'])
                        attention_masks[i, num_examples, j, k] = np.array(tmp_tok['attention_mask'])
                        tmp_label_mask = np.zeros_like(tokens[i, num_examples, j, k])
                        
                        idx = len(tokenized_label)
                        tmp_label_mask[-idx:] = 1
                        label_masks[i, num_examples, j, k] = tmp_label_mask
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
            for num_examples in range(_k+1):
                for j in range(max_num_classes):
                    for k in range(total_examples):
                        curr_length = len(tokens[i, num_examples, j, k])
                        if curr_length > max_len:
                            max_len = curr_length

        padded_tokens = np.full((n, _k+1, max_num_classes, total_examples, max_len), pad_token_id)
        padded_attention_mask = np.full((n, _k+1, max_num_classes, total_examples, max_len), 0)
        padded_label_mask = np.full((n, _k+1, max_num_classes, total_examples, max_len), 0)
                
        for i in range(n):
            print(f"padding: {i}/{n}", end="\r")
            for num_examples in range(_k+1):
                for j in range(max_num_classes):
                    for k in range(total_examples):
                        padded_tokens[i, num_examples, j, k, :len(tokens[i, num_examples, j, k])] = tokens[i, num_examples, j, k]
                        padded_attention_mask[i, num_examples, j, k, :len(attention_masks[i, num_examples, j, k])] = attention_masks[i, num_examples, j, k]
                        padded_label_mask[i, num_examples, j, k, :len(label_masks[i, num_examples, j, k])] = label_masks[i, num_examples, j, k]
        # print("***************")
        tokenized_dataset = {
            'input_ids': torch.from_numpy(padded_tokens), # (n, num_classes, total_examples, maxlen)
            'attention_mask': torch.from_numpy(padded_attention_mask), # (n, num_classes, total_examples, maxlen)
            'label_mask': torch.from_numpy(padded_label_mask), # (n, num_classes, total_examples, maxlen)
            'labels': torch.from_numpy(labels), # (n)
        }
        # for key, val in tokenized_dataset.items():
        #     torch.save(val, f'{preprocess_path}-{key}.pt')

        return tokenized_dataset
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, batch_size=len(raw_dataset[labelfield]))
    columns_to_remove = raw_dataset.column_names

    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)

    logging.info(tokenized_dataset)
    tokenized_dataset.set_format("torch")
    return tokenized_dataset
