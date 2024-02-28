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

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding, HfArgumentParser, BitsAndBytesConfig
from datasets import load_dataset, Dataset

from config_channel_ablate import TASK2LABELSTRINGS as TASK2ABLATELABELSTRINGS
from config_fewshot_new import TASK2LABELSTRINGS, EXAMPLEFORMAT2ENTAIL, EXAMPLEFORMAT2NOTENTAIL, EXAMPLEFORMAT_SPACE2ENTAIL, EXAMPLEFORMAT_SPACE2NOTENTAIL#, EXAMPLEFORMAT2, EXAMPLEFORMAT2_SPACE

from dataset_loaders import TASK2LOADER, TOKEN
# from cc import cc
import logging

def get_tokenized_dataset_mcq_context_fewshot(raw_dataset, raw_train_dataset, tokenizer, question="sentence1", passage="passage", labelfield="answer", label2id=None, _task="", _k=10, _num_sets=8, possible_labels_string="", create=False, preprocess_path="", few_shot_seed=2024):
    def preprocess_function(examples): # would need to keep the prem and hyp separate, or just get the mask here
        # if os.path.isfile(f"{preprocess_path}-input_ids.pt") and not create:
        #     print("loading...")
        #     tokenized_dataset = {"input_ids": None, "attention_mask": None, "label_mask": None, "labels": None}
        #     for key in tokenized_dataset.keys():
        #         tokenized_dataset[key] = torch.load(f'{preprocess_path}-{key}.pt')
        #     return tokenized_dataset
        np.random.seed(few_shot_seed)   
        examples_questions = examples[question]
        n = len(examples_questions)
        choices = possible_labels_string.split(",")
        examples_choices = [choices] * n
        examples_questions = examples[question]
        examples_passages = examples[passage]
        possible_labels = possible_labels_string.split(",")
        
        pad_token_id = tokenizer.pad_token_id
        template = TASK2LABELSTRINGS[_task][0] # (num classes by num_label_strings)
        choice_template = TASK2LABELSTRINGS[_task][1]
        
        labels = np.array([possible_labels.index(f"{examples[labelfield][i]}") for i in range(n)])
        num_classes = len(possible_labels)

        all_choices = np.array(examples_choices)

        # prepare fewshot examples
        few_shot_examples = np.empty((_k+1, _num_sets), dtype=object)
        train_data = raw_train_dataset
        train_questions = np.array(train_data[question])
        train_passages = np.array(train_data[passage])
        train_choices = np.array([possible_labels] * len(train_questions))
        train_labels = np.array([possible_labels.index(f"{train_data[labelfield][i]}") for i in range(len(train_questions))])

        for i in range(_num_sets):
            # index into and get the k strings
            idxs = np.random.choice(range(0, len(train_questions)), _k, replace=False)
            few_shot = {}
            few_shot["passage"] = train_passages[idxs]
            few_shot["question"] = train_questions[idxs]
            few_shot["choices"] = train_choices[idxs, :]
            few_shot["label"] = train_labels[idxs]

            few_shot_strs = np.empty((_k), dtype=object)
            # format all k strings 
            for j in range(_k):
                tmp_choices = ""
                for choice in range(num_classes):
                    tmp_choices += choice_template.format(label=choice+1, choice=few_shot["choices"][j][choice])
                lab = possible_labels[few_shot["label"][j]]
                few_shot_strs[j] = template.format(passage=few_shot["passage"][j], question=few_shot["question"][j], choices=tmp_choices, label=lab)
            
            # compile all the strings from 1 to k and concant ino one string (so total k strings)
            for num_examples in range(_k+1):
                example_idxs = np.random.choice(range(0, _k), num_examples, replace=False)
                few_shot_strs_num_examples = few_shot_strs[example_idxs]
                tmp_str = ""    
                for k in range(num_examples):
                    tmp_str += few_shot_strs_num_examples[k] + "\n\n"
                few_shot_examples[num_examples, i] = tmp_str

        total_examples = _num_sets
        tokens = np.full((n, _k+1, num_classes, total_examples), None)
        attention_masks = np.full((n, _k+1, num_classes, total_examples), None)
        label_masks = np.full((n, _k+1, num_classes, total_examples), None)
        tok_x = tokenizer.encode("x", add_special_tokens=False)

        for i in range(n):
            quoted_passage = examples_passages[i]
            quoted_question = examples_questions[i]
            tmp_choices = ""
            for choice in range(num_classes):
                tmp_choices += choice_template.format(label=choice+1, choice=all_choices[i, choice])
            quoted_choices = tmp_choices
            for num_examples in range(_k+1):
                for j in range(num_classes):
                    label = possible_labels[j]
                    tokenized_label = tokenizer.encode("x " + label, add_special_tokens=False)[len(tok_x):]
                    for k in range(total_examples):
                        string = few_shot_examples[num_examples, k] + template.format(passage=quoted_passage, question=quoted_question, choices=quoted_choices, label=label)
                        
                        tmp_tok = tokenizer(string)
                        tokens[i, num_examples, j, k] = np.array(tmp_tok['input_ids'])
                        attention_masks[i, num_examples, j, k] = np.array(tmp_tok['attention_mask'])
                        tmp_label_mask = np.zeros_like(tokens[i, num_examples, j, k])
                        
                        idx = len(tokenized_label)
                        tmp_label_mask[-idx:] = 1
                        label_masks[i, num_examples, j, k] = tmp_label_mask

                        tmp = np.array(label_masks[i, num_examples, j, k]) * np.array(tokens[i, num_examples, j, k])
                        # print(tokenizer.decode(tmp[tmp != 0]))

                        # print("{" + string+"}")
                        # print()
                        # input()

    
        max_len = 0
        
        for i in range(n):
            for num_examples in range(_k+1):
                for j in range(num_classes):
                    for k in range(total_examples):
                        curr_length = len(tokens[i, num_examples, j, k])
                        if curr_length > max_len:
                            max_len = curr_length
        # print(")()", max_len)
        padded_tokens = np.full((n, _k+1, num_classes, total_examples, max_len), pad_token_id)
        padded_attention_mask = np.full((n, _k+1, num_classes, total_examples, max_len), 0)
        padded_label_mask = np.full((n, _k+1, num_classes, total_examples, max_len), 0)
                
        for i in range(n):
            for num_examples in range(_k+1):
                for j in range(num_classes):
                    for k in range(total_examples):
                        padded_tokens[i, num_examples, j, k, :len(tokens[i, num_examples, j, k])] = tokens[i, num_examples, j, k]
                        padded_attention_mask[i, num_examples, j, k, :len(attention_masks[i, num_examples, j, k])] = attention_masks[i, num_examples, j, k]
                        padded_label_mask[i, num_examples, j, k, :len(label_masks[i, num_examples, j, k])] = label_masks[i, num_examples, j, k]
        # print("%%%%%%%")
        # print(labels[:10])
        # print(padded_tokens[:10, 0, 0, :])
        tokenized_dataset = {
            'input_ids': torch.from_numpy(padded_tokens), # (n, num_classes, total_examples, maxlen)
            'attention_mask': torch.from_numpy(padded_attention_mask), # (n, num_classes, total_examples, maxlen)
            'label_mask': torch.from_numpy(padded_label_mask), # (n, num_classes, total_examples, maxlen)
            'labels': torch.from_numpy(labels), # (n)
        }
        # print("***************")
        # for key, val in tokenized_dataset.items():
        #     print(key, ":", val.shape)


        return tokenized_dataset
    # print("777777777")
    # print(len(raw_dataset[labelfield]))
    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True, batch_size=len(raw_dataset[labelfield]))
    columns_to_remove = raw_dataset.column_names
    if label2id is None:
        columns_to_remove.remove(labelfield)
    tokenized_dataset = tokenized_dataset.remove_columns(columns_to_remove)
    logging.info(tokenized_dataset)
    tokenized_dataset.set_format("torch")

    return tokenized_dataset
