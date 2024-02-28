import os
import numpy as np

from datasets import Dataset, load_dataset, get_dataset_config_names
def get_mmlu(split="validation", num=20):
    np.random.seed(2024)   
    vals = {
        'input': [],
        'A': [],
        'B': [],
        'C': [],
        'D': [], 
        'target': [],
        'domain': []
    }
    for count, config in enumerate(get_dataset_config_names("lukaemon/mmlu")):
        tmp = load_dataset("lukaemon/mmlu", config)[split][:num]
        size = len(tmp['input'])
        for i in range(size):
            for key in vals.keys():
                if key != 'domain':
                    vals[key].append(tmp[key][i])
                else:
                    vals[key].append(config.replace('_', ' '))
    return Dataset.from_dict(vals)
