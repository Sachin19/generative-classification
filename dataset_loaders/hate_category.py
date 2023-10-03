import os
import numpy as np

from datasets import Dataset, load_dataset

def get_evaluation_set(filename, target_group):
    dataset = load_dataset("json", data_files=f"/projects/tir5/users/sachink/generative-classifiers/2023/datasets/hate-features/{filename}", split="train", cache_dir="/projects/tir5/users/sachink/generative-classifiers/2023/datasets/").filter(lambda example: example["fold"] == "test")
    test_lines = get_test_lines(dataset, target_group)
    test_lines = list(zip(*test_lines))
    return Dataset.from_dict({'text': test_lines[0], 'label': test_lines[1]})


def get_test_lines(dataset, target_group):
    # only train set, manually split 20% data as test
    lines = map_hf_dataset_to_list(dataset, target_group)

    return lines

def map_hf_dataset_to_list(hf_dataset, target_group):
    lines = []
    print(target_group)
    for datapoint in hf_dataset:
        # line[0]: input; line[1]: output
        if target_group is None or target_group in datapoint['target_groups']:
            lines.append((datapoint["text"].strip(), int(datapoint["hate"])))
    return lines
