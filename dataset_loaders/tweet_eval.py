import os
import numpy as np

from datasets import Dataset, load_dataset

def get_evaluation_set(subset):
    dataset =load_dataset("tweet_eval", subset)
    _, test_lines = get_train_test_lines(dataset)
    test_lines = list(zip(*test_lines))
    return Dataset.from_dict({'text': test_lines[0], 'label': test_lines[1]})


def get_train_test_lines(dataset):
    # only train set, manually split 20% data as test

    test_lines = map_hf_dataset_to_list(dataset, "test")

    # np.random.seed(42)
    # np.random.shuffle(lines)
    
    # n = len(lines)

    # train_lines = lines[:int(0.8*n)]
    # test_lines = lines[int(0.8*n):]

    return None, test_lines

def map_hf_dataset_to_list(hf_dataset, split_name):
    lines = []
    for datapoint in hf_dataset[split_name]:
        # line[0]: input; line[1]: output
        lines.append((datapoint["text"].strip(), datapoint["label"]))
    return lines

#dimensions = ["directed_vs_generalized", "disability", "gender", "national_origin", "race", "religion", "sexual_orientation"]