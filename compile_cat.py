import sys

path = sys.argv[1]
num_seeds = sys.argv[2]
file_name = sys.argv[3]
save_dir = sys.argv[4]

results = []
for i in range(1, num_seeds+1):
    with open(path + f"/{i}/" +filename, "r") as f:
        results.append([float(i) for i in f.readline()])

arr = np.array(results)
results = np.mean(arr, axis=0)
stds = np.std(arr, axis=0)
with open(save_dir, 'w') as f:
    f.write(" ".join(results.astype(str)) + "\n")
    f.write(" ".join(results.astype(str)))