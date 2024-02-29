import numpy as np
import sys 
from matplotlib import pyplot as plt

# python calc_cat.py original_path cat_path save_path file_name neutral_idx num_runs 
# print()
# print("**********")
# print(sys.argv)
original_path = sys.argv[1]
cat_path = sys.argv[2]
save_path = sys.argv[2]
file_name = sys.argv[3]
neutral_idx = int(sys.argv[4])
num_runs = int(sys.argv[5])
save_file = sys.argv[6]


def compile(path, num_runs):
    all_preds = []
    for i in range(num_runs): # for each run
        preds_for_one_file = []
        with open(f"{path}/run-{i}_{file_name}", 'r') as f:
            content = f.readlines()
            for line in content: 
                vals = line.strip().split()
                if len(vals) > 1:
                    preds = [int(val) for val in vals[1:]]
                    preds_for_one_file.append(preds)

        all_preds.append(preds_for_one_file)
    # print(all_preds)
    return np.swapaxes(np.array(all_preds), 0, 1)
    
preds = compile(original_path, num_runs) # n, num_runs, _k
mod_preds = compile(cat_path, num_runs) # same shape
# print(preds)
# print("\n\n\n")
# print(mod_preds)
# print(preds.shape)
n, num_sets, _k = preds.shape
num_corr = np.full((num_sets, num_sets, _k), 0)
total = np.full((num_sets, num_sets, _k), 1e-12)
for i in range(n):
    for j in range(num_sets):
        for j1 in range(num_sets):
            for k in range(_k):
                pred = preds[i, j, k] 
                mod_pred = mod_preds[i, j1, k]
                if pred != neutral_idx:
                    total[j, j1, k] += 1
                    if mod_pred != pred:
                        num_corr[j, j1, k] += 1
cats = np.average(num_corr / total, axis=(0, 1))

with open(save_path+f"/{save_file}", 'w') as f:
    f.write(" ".join(cats.astype(str)) + "\n")

# plt.plot(range(len(cats)), cats)

# plt.xlabel("num fewshot examples")
# plt.ylabel("CAT Score")
