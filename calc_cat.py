import numpy as np
import sys 
from matplotlib import pyplot as plt

# python calc_cat.py original_path cat_path save_path file_name neutral_idx num_runs 
neutral_idx = int(sys.argv[5])
num_runs = int(sys.argv[6])
print(sys.argv)

def compile(path, num_runs):
    all_preds = []
    for i in range(num_runs): # for each run
        preds_for_one_file = []
        with open(f"{path}/run-{i}_{sys.argv[4]}.txt", 'r') as f:
            content = f.readlines()
            for line in content: 
                vals = line.strip().split()
                if len(vals) > 1:
                    preds = [int(val) for val in vals[1:]]
                    preds_for_one_file.append(preds)

        all_preds.append(preds_for_one_file)
    # print(all_preds)
    return np.swapaxes(np.array(all_preds), 0, 1)
    
preds = compile(sys.argv[1], num_runs) # n, num_runs, _k
mod_preds = compile(sys.argv[2], num_runs) # same shape

n, num_sets, _k = preds.shape
num_corr = np.full((_k), 0)
total = np.full((_k), 1e-10)
for i in range(n):
    for k in range(_k):
        pred = np.bincount(preds[i, :, k]).argmax() # get the argmax over all runs i.e. the actual predicted value
        mod_pred = np.bincount(mod_preds[i, :, k]).argmax()
        if pred != neutral_idx:
            total[k] += 1
            if mod_pred != pred:
                num_corr[k] += 1

print(total)
print(num_corr)
cats = num_corr / total
print(cats)
with open(sys.argv[3], 'w') as f:
    f.write(" ".join(cats.astype(str)) + "\n")

# plt.plot(range(len(cats)), cats)

# plt.xlabel("num fewshot examples")
# plt.ylabel("CAT Score")
# plt.savefig(sys.argv[3])