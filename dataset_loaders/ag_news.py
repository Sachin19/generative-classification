import os
import numpy as np

from datasets import Dataset, load_dataset

def get_evaluation_set(file_path):
    # dataset =load_dataset("tweet_eval", subset)
    dataset = load_dataset("csv", data_files=file_path)
    _, test_lines = get_train_test_lines(dataset)
    test_lines = list(zip(*test_lines))
    return Dataset.from_dict({'text': test_lines[0], 'label': test_lines[1]})


def get_train_test_lines(dataset):
    # only train set, manually split 20% data as test

    test_lines = map_hf_dataset_to_list(dataset, "train") # this is actually test, load_dataset names it weird
    return None, test_lines

def map_hf_dataset_to_list(hf_dataset, split_name):
    lines = []
    for datapoint in hf_dataset[split_name]:
        # line[0]: input; line[1]: output
        text = datapoint["Title"] + ". " + datapoint['Description']
        text = text.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") # some basic cleaning 
        label = datapoint['Class Index'] - 1
    
        lines.append((text, label))
    return lines

#dimensions = ["directed_vs_generalized", "disability", "gender", "national_origin", "race", "religion", "sexual_orientation"]