import numpy as np
import sys
import json
from matplotlib import pyplot as plt

json_file = sys.argv[1]

def matrix_to_precision(conf: str):
    # row is truth, col is predicted
    rows = conf.split("\n ")
    # mat = np.zeros((len(rows), len(rows)))
    mat = []
    # print(rows)
    rows[0] = rows[0][1:]
    rows[-1] = rows[-1][:-1]
    for row in rows:
        mat.append([int(i) for i in row[1:-1].split()])
    mat = np.array(mat)
    precision = np.diag(mat)/ (np.sum(mat, axis=1) + 1e-10)
    return precision

all_confs = []
with open(sys.argv[1], "r") as f:
    for line in f:
        items = json.loads(line)
        all_confs.append(items["confusion_matrix_arithmetic"])

all_precs = [[matrix_to_precision(conf) for conf in confs] for confs in all_confs]

avg = []
for i in range(len(all_precs[0])):
    avg_prec = np.zeros_like(all_precs[0][0])
    for j in range(len(all_precs)):
        avg_prec += all_precs[j][i]
    avg_prec /= len(all_precs)
    avg.append(avg_prec)
avg = np.array(avg)
for i in range(avg.shape[1]):
    plt.plot(range(avg.shape[0]), avg[:, i], label=f"class{i}")
plt.legend()
plt.savefig(sys.argv[2])

matrix_to_precision("[[13 10  0]\n [ 9 19  0]\n [ 5  0  0]]")
