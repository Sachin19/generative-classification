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

def get_tokenized_dataset_nli_fewshot_custom(raw_dataset, raw_train_dataset, tokenizer, textfield1="sentence1", textfield2="sentence2", labelfield="label", label2id=None, task="", _k=10, _num_sets=8, _label_names="", create=False, preprocess_path="", few_shot_seed=2024):
    def preprocess_function(examples):  
        np.random.seed(few_shot_seed)          
        print("creating")
        print("****************")
        pad_token_id = tokenizer.pad_token_id
        template = TASK2LABELSTRINGS[task]
        label_names = _label_names.split(",")
        tok_x = tokenizer.encode("x", add_special_tokens=False)
        # tok_labels = [tokenizer.encode("x " + label_name, add_special_tokens=False)[len(tok_x):] for label_name in label_names]
        # tok_label_lens = [len(tok_label) for tok_label in tok_labels]
        num_classes = len(label_names)

        premises = np.array(examples[textfield1])
        hypothesis = np.array(examples[textfield2])
        labels = np.array(examples[labelfield])
        
        n = len(labels)

        # get few shot examples
        few_shot_examples = np.empty((num_classes, (_k)+1, _num_sets), dtype=object)
        train_data = raw_train_dataset
        train_premises = np.array(train_data[textfield1])
        train_hypothesis = np.array(train_data[textfield2])
        train_labels = np.array(train_data[labelfield])
        premises_split_by_label = []
        hypothesis_split_by_label = []
        for _class in range(num_classes):
            premises_split_by_label.append(np.array([train_premises[i] for i in range(len(train_premises)) if train_labels[i] == _class]))
            hypothesis_split_by_label.append(np.array([train_hypothesis[i] for i in range(len(train_hypothesis)) if train_labels[i] == _class]))
        print("OK1")
        print(len(premises_split_by_label), len(premises_split_by_label[0]))
        print(len(hypothesis_split_by_label), len(hypothesis_split_by_label[0]))
        # input()
        # for i in range(num_classes):
        #     print(label_names[i])
        #     for j in range(4):
        #         print(premises_split_by_label[i][j], ",", hypothesis_split_by_label[i][j])
        #         input()
        #         # print()
        #     print()
        # print(premises_split_by_label[0][2])
        # print(hypothesis_split_by_label[0][2])
        # input()
        for i in range(_num_sets):
            idxs = [np.random.choice(range(0, len(premises_split_by_label[l])), 2*_k, replace=False) for l in range(num_classes)]
            few_shot = {}
            few_shot["premise"] = [premises_split_by_label[l][idxs[l]] for l in range(num_classes)]
            few_shot["hypothesis"] = [hypothesis_split_by_label[l][idxs[l]] for l in range(num_classes)]
            # few_shot["label"] = train_labels[idxs]
            few_shot_strs = np.empty((num_classes, 2*_k), dtype=object)
            # print("OK2")
            for l in range(num_classes):
                for j in range(2*_k):
                    few_shot_strs[l, j] = ""
                    few_shot_strs[l, j] += template[l].format(premise="\""+few_shot["premise"][l][j]+"\"", hypothesis="\""+few_shot["hypothesis"][l][j]+"\"") + "\n\n"
            # print(few_shot_strs)
            # input()
            for _class in range(num_classes):
                for num_examples in range(_k+1):
                    example_idxs = np.random.choice(range(0, 2*_k), 2 * num_examples, replace=False)
                    few_shot_strs_num_examples = few_shot_strs[_class, example_idxs]

                    tmp_str = ""
                        
                    for k in range(2 * num_examples):
                        tmp_str += few_shot_strs_num_examples[k]
                    # print(tmp_str)
                    # input()
                    few_shot_examples[_class, num_examples, i] = tmp_str
        # print("OK3")
        # for i in range(_k+1):
        #     print(i)
        #     for j in range(_num_sets):
        #         print(few_shot_examples[0, i, j])
        #         print("******")
        #         print(few_shot_examples[1, i, j])
        #         input()
        #     print()
        # # print(few_shot_examples)
        # print(len(few_shot_examples), len(few_shot_examples[0]))
        tokens = np.full((n, _k+1, num_classes, _num_sets), None)
        # cf_tokens = np.full((n, _k+1, num_classes, _num_sets, num_cf), None)
        attention_masks = np.full((n, _k+1, num_classes, _num_sets), None)
        # cf_attention_masks = np.full((n, _k+1, num_classes, _num_sets, num_cf), None)
        label_masks = np.full((n, _k+1, num_classes, _num_sets), None)
        # cf_label_masks = np.full((n, _k+1, num_classes, _num_sets, num_cf), None)
        
        for i in range(n):
            premise = "\"" + premises[i] + "\""
            hyp = "\"" + hypothesis[i] + "\"" + tokenizer.eos_token
            tok_hyp = tokenizer.encode("x " + hyp, add_special_tokens=False)[len(tok_x):]
            for num_examples in range(_k+1):
                for j in range(num_classes):
                    label = label_names[j]
                     
                    for k in range(_num_sets):
                        string = few_shot_examples[j, num_examples, k] + template[j].format(premise=premise, hypothesis=hyp)
                        # print(string)
                        # print()
                        tmp_tok = tokenizer(string)
                        tokens[i, num_examples, j, k] = np.array(tmp_tok['input_ids'])
                        attention_masks[i, num_examples, j, k] = np.array(tmp_tok['attention_mask'])
                        tmp_label_mask = np.zeros_like(tokens[i, num_examples, j, k])
                        
                        idx = len(tok_hyp)
                        tmp_label_mask[-idx:] = 1
                        label_masks[i, num_examples, j, k] = tmp_label_mask
                        # assert np.sum(label_masks[i, num_examples, j, k]) == tok_label_lens[j]
                        tmp_prd = (label_masks[i, num_examples, j, k] * tokens[i, num_examples, j, k])
                        # print(tokenizer.decode(tmp_prd[tmp_prd != 0]))
                        # input()
                    # print("\n\n")

                        ## do the cf string now
                        # for c in range(num_cf):
                        #     cf_string = few_shot_examples[num_examples, k] + template.format(premise=cfs[c], hypothesis=cfs[c], label=label)
                        #     # print(cf_string)

                        #     cf_tmp_tok = tokenizer(cf_string)
                        #     cf_tokens[i, num_examples, j, k, c] = np.array(cf_tmp_tok['input_ids'])
                        #     cf_attention_masks[i, num_examples, j, k, c] = np.array(cf_tmp_tok['attention_mask'])
                        #     cf_tmp_label_mask = np.zeros_like(cf_tokens[i, num_examples, j, k, c])

                        #     idx = len(tokenized_label)
                        #     cf_tmp_label_mask[-idx:] = 1
                        #     cf_label_masks[i, num_examples, j, k, c] = cf_tmp_label_mask

        max_len = 0 # it's safe to assume that the max_len for the actual strings will be greater than the max_len for the cf, as we are replacing the hypothesis and premise by one word
        print("past assertions")
        # input()
        for i in range(n):
            for num_examples in range(_k+1):
                for j in range(num_classes):
                    for k in range(_num_sets):
                        curr_length = len(tokens[i, num_examples, j, k])
                        if curr_length > max_len:
                            max_len = curr_length
        
        padded_tokens = np.full((n, _k+1, num_classes, _num_sets, max_len), pad_token_id)
        padded_attention_mask = np.full((n, _k+1, num_classes, _num_sets, max_len), 0)
        padded_label_mask = np.full((n, _k+1, num_classes, _num_sets, max_len), 0)

                
        for i in range(n):
            for num_examples in range(_k+1):
                for j in range(num_classes):
                    for k in range(_num_sets):
                        padded_tokens[i, num_examples, j, k, :len(tokens[i, num_examples, j, k])] = tokens[i, num_examples, j, k]
                        padded_attention_mask[i, num_examples, j, k, :len(attention_masks[i, num_examples, j, k])] = attention_masks[i, num_examples, j, k]
                        padded_label_mask[i, num_examples, j, k, :len(label_masks[i, num_examples, j, k])] = label_masks[i, num_examples, j, k]

                        # test
                        test_prd = padded_label_mask[i, num_examples, j, k, :] * padded_tokens[i, num_examples, j, k, :]

                        # assert (test_prd[test_prd != 0] == tok_labels[j]).all()
                        # print(tokenizer.decode(test_prd[test_prd != 0]))
        print("passed assertions")
        # print("%%%%%%%")
        # print(labels[:10])
        # print(padded_tokens[:10, 0, 0, :])
        tokenized_dataset = {
            'input_ids': torch.from_numpy(padded_tokens),
            'attention_mask': torch.from_numpy(padded_attention_mask),
            'label_mask': torch.from_numpy(padded_label_mask),
            'labels': torch.from_numpy(labels), # (n)
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
