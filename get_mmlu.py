import os
import numpy as np

from datasets import Dataset, load_dataset, get_dataset_config_names
def get_mmlu(split="validation", num=10):
    num = 10
    np.random.seed(2024)   
    vals = {
        'input': [],
        'choices': [],
        'target': [],
        'domain': []
    }
    for count, config in enumerate(get_dataset_config_names("lukaemon/mmlu")):
        if config == "all":
            continue
        if "train" not in config:
            tmp = load_dataset("lukaemon/mmlu", config)[split]
            # print(config)
            # input()
            size = len(tmp['input'])
            for i in range(min(num, size)):
                choice = []
                for key in ['input', 'A', 'B', 'C', 'D', 'target', 'domain']:
                    if len(key) == 1:
                        choice.append(tmp[key][i])
                    elif key != 'domain':
                        vals[key].append(tmp[key][i])
                    else:
                        vals[key].append(config.replace('_', ' '))
                vals['choices'].append(choice)
        # if count == 2:
        #     break
    return Dataset.from_dict(vals)
