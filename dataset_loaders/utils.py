import numpy as np

def down_size_dataset(lines, n):
    np.random.seed(100)

    new_lines = []
        
    # count labels
    label_indices = {}
    for i, (_, label) in enumerate(lines):
        if label in label_indices:
            label_indices[label].append(i) 
        else:
            label_indices[label] = [i]
    
    n_labels =  len(label_indices.keys())
    n_entry = sum([len(v) for k, v in label_indices.items()])
    n_per_label = {k: int(n / n_entry * len(v)) for k, v in label_indices.items()}
    
    # fill up remainding test inputs
    diff = n - sum(n_per_label.values())
    if diff > 0:
        for i, (k, v) in enumerate(n_per_label.items()):
            if i >= diff:
                break
            n_per_label[k] +=1 
    assert n - sum(n_per_label.values()) == 0

    selected_indices = []
    # randomly select indices
    for label, indices in label_indices.items():
        selected_indices.extend(np.random.choice(indices, n_per_label[label], replace=False))

    new_lines = [dp for i, dp in enumerate(lines) if i in selected_indices]
    np.random.shuffle(new_lines)

    return new_lines