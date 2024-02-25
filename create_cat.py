import numpy as np
from datasets import load_dataset, Dataset

def create_cat(dataset: str, data_dir: str, split: str, text1, text2, labels, seed=2024): 
    """
        seed: random seed for generation
        dataset: dataset to choose from
        data_dir: subset
        split: train, validation, test
    """
    print("catting")
    np.random.seed(seed)
    raw_dataset = load_dataset(dataset, data_dir, split=split, cache_dir="datasets")
    new_dataset = {
        text1: {},
        text2: {},
        labels: {}
    }
    new_dataset[text1] = [None] * len(raw_dataset[text1])
    for i in range(len(new_dataset[text1])):
        idx = i
        while idx == i:
            idx = np.random.randint(0, len(new_dataset[text1]))
        new_dataset[text1][i] = raw_dataset[text1][idx]
        

    # np.random.shuffle(new_dataset[text1])
    new_dataset[text2], new_dataset[labels] = raw_dataset[text2], raw_dataset[labels]
    # print(len(new_dataset[text2]))
    return Dataset.from_dict(new_dataset)
