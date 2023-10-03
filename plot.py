import json
import numpy as np
import sys 
from matplotlib import pyplot as plt

if len(sys.argv) >= 4:
    prefix=sys.argv[3]
    print(prefix)
    if prefix == "channel++_":
        sys.exit(0)
    elif prefix == "channel_":
        prefix=""
else:
    prefix=""

with open(sys.argv[1]) as f:
    y_arithmetic = []
    y_geometric = []
    y_harmonic = []
    x = None
    c = 0
    for line in f:
        items = json.loads(line)
        if x is None:
            x = items['k']
        y_arithmetic.append(items[prefix+'metric_arithmetic'])
        y_geometric.append(items[prefix+'metric_geometric'])
        y_harmonic.append(items[prefix+'metric_harmonic'])
    y_arithmeticmean = np.mean(y_arithmetic, axis=0)
    y_geometricmean = np.mean(y_geometric, axis=0)
    y_harmonicmean = np.mean(y_harmonic, axis=0)

    y_arithmeticstd = np.std(y_arithmetic, axis=0)
    y_geometricstd = np.std(y_geometric, axis=0)
    y_harmonicstd = np.std(y_harmonic, axis=0)

plt.errorbar(x, y_arithmeticmean, y_arithmeticstd, color="red", label="arithmetic")
plt.errorbar(x, y_geometricmean, y_geometricstd, color="blue", label="geometric")
plt.errorbar(x, y_harmonicmean, y_harmonicstd, color="green", label="harmonic")
plt.legend(loc="best")
plt.savefig(sys.argv[2])