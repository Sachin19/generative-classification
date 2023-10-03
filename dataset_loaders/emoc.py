import os
import numpy as np

from datasets import Dataset, load_dataset

def get_evaluation_set(subset, few_shot_k=0, few_shot_seed=0):
    dataset = load_dataset("emo")# cache_dir="/projects/tir5/users/sachink/generative-classifiers/2023/datasets/")
    few_shot_examples, test_lines = get_train_test_lines(dataset, int(few_shot_k), int(few_shot_seed))
    test_lines = list(zip(*test_lines))

    # input(len(test_lines))
    # if few_shot_k > 0:
    #     np.set_seed(few_shot_seed)
    #     np.random.choice(train_lines, )
    if few_shot_k > 0:
        return (Dataset.from_dict({'text': test_lines[0], 'label': test_lines[1]}), few_shot_examples)
    else:
        return Dataset.from_dict({'text': test_lines[0], 'label': test_lines[1]})

def get_train_test_lines(dataset, k, seed):
    # only train set, manually split 20% data as test
    if k > 0:
        train_lines = map_hf_dataset_to_list(dataset, "train")
        np.random.seed(seed)
        np.random.shuffle(train_lines)
        few_shot_examples = train_lines[:k]
    else:
        few_shot_examples = []
    test_lines = map_hf_dataset_to_list(dataset, "test")

    # np.random.seed(42)
    # np.random.shuffle(lines)
    
    # n = len(lines)

    # train_lines = lines[:int(0.8*n)]
    # test_lines = lines[int(0.8*n):]

    return few_shot_examples, test_lines

def map_hf_dataset_to_list(hf_dataset, split_name):
    lines = []
    for datapoint in hf_dataset[split_name]:
        # line[0]: input; line[1]: output
        lines.append((datapoint["text"].strip(), datapoint["label"]))
    return lines