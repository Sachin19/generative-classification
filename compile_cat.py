import sys
import numpy as np
from matplotlib import pyplot as plt

# print(sys.argv)
cat_res_path = sys.argv[1]
num_seeds = int(sys.argv[2])
file_name = sys.argv[5]
save_path = sys.argv[3]
plt_path = sys.argv[4]

results = []
for i in range(1, num_seeds+1):
    with open(cat_res_path + f"/{i}/" +file_name, "r") as f:
        results.append([float(i) for i in f.readline().split()])

arr = np.array(results)
results = np.mean(arr, axis=0)
plus_minus = np.full_like(results.astype(str), "+-")
stds = np.std(arr, axis=0)

final = [str(results[i]) + plus_minus[i] + str(stds[i]) for  i in range(results.shape[0])]
# print(final)
# print(results)
# print(stds)
with open(save_path, 'w') as f:
    f.write(",".join(final))

plt.errorbar(range(len(final)), results, stds)
plt.savefig(plt_path)