import os
import numpy as np

from dataset_loaders.utils import down_size_dataset
from datasets import Dataset, load_dataset

def get_evaluation_set():
    dataset =load_dataset("yelp_review_full")
    _, test_lines = get_train_test_lines(dataset)
    test_lines = list(zip(*test_lines))
    return Dataset.from_dict({'text': test_lines[0], 'label': test_lines[1]})


def get_train_test_lines(dataset):
    # only train set, manually split 20% data as test

    lines = map_hf_dataset_to_list(dataset, "test")
    # using 20% of test cases, otherwise it's too slow to do evaluation

    test_lines = down_size_dataset(lines, 2000)

    return None, test_lines
    
def map_hf_dataset_to_list(hf_dataset, split_name):
    lines = []
    for datapoint in hf_dataset[split_name]:
        # line[0]: input; line[1]: output
        lines.append((datapoint["text"].strip(), datapoint["label"]))
    return lines

#dimensions = ["directed_vs_generalized", "disability", "gender", "national_origin", "race", "religion", "sexual_orientation"]