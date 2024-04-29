import sys
import numpy as np
from matplotlib import pyplot as plt
import pickle

# print(sys.argv)
cat_res_path = sys.argv[1]
num_seeds = int(sys.argv[2])
file_name = sys.argv[5]
save_path = sys.argv[3]
plt_path = sys.argv[4]

results = []
counts = []
for i in range(1, num_seeds+1):
    with open(cat_res_path + f"/{i}/" +file_name, "rb") as f:
        results.append(pickle.load(f))
    # print("com")
    # print(cat_res_path + f"/{i}/count-" +file_name)
    with open(cat_res_path + f"/{i}/count-" +file_name, "rb") as f:
        counts.append(pickle.load(f))
        # results.append([float(i) for i in f.readline().split()])

arr = np.array(results)
c_arr = np.array(counts)
# print(arr)
# print(arr.shape)
results = np.mean(arr, axis=(0, 1))
c_results = np.mean(counts, axis=(0, 1))
# print(results)
# print(results.shape)
plus_minus = np.full_like(results.astype(str), "+-")
stds = np.std(arr, axis=(0, 1))
c_stds = np.std(c_arr, axis=(0, 1))

final = [str(results[i]) + plus_minus[i] + str(stds[i]) for  i in range(results.shape[0])]
c_final = [str(c_results[i]) + plus_minus[i] + str(c_stds[i]) for  i in range(c_results.shape[0])]
# print(final)
# print(results)
# print(stds)
with open(save_path, 'w') as f:
    f.write(",".join(final))
    f.write("\n")
    f.write(",".join(c_final))

plt.errorbar(range(len(final)), results, stds)
plt.savefig(plt_path)