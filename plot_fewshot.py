import json
import numpy as np
import sys 
from matplotlib import pyplot as plt

# print("yes!!")
if len(sys.argv) >= 4:
    title=sys.argv[3]
    # print(prefix)
    # if prefix == "channel++_":
    #     sys.exit(0)
    # elif prefix == "channel_":
    #     prefix=""
else:
    title=""
colors = ["red", "blue", "green", "black"]
i = 0
settings = [""]
if not ("hGivenP" in title or "pGivenH" in title):
    settings.append("cc_")


for setting in settings:
    for m in ['accuracy', 'f1']:
        with open(sys.argv[1]) as f:
            # y_arithmetic = {}
            # y_geometric = {}
            # y_harmonic = {}
            # print(m)
            y = []
            # y_geometric = []
            # y_harmonic = []
            x = None
            c = 0
            for line in f:
                items = json.loads(line)
                if x is None:
                    x = items['k']
                y.append(items[setting+m])
                # y_geometric.append(items[m+'_geometric'])
                # y_harmonic.append(items[m+'_harmonic'])

            y_mean = np.mean(y, axis=0)
            # y_geometricmean = np.mean(y_geometric, axis=0)
            # y_harmonicmean = np.mean(y_harmonic, axis=0)

            y_std = np.std(y, axis=0)
            # y_geometricstd = np.std(y_geometric, axis=0)
            # y_harmonicstd = np.std(y_harmonic, axis=0)
        

            # print(y)
            # print(y_geometric)
            # print(x)
            plt.errorbar([1*i for i in x], y_mean, y_std, color=colors[i], label=setting+m)
            i += 1
            # print("1")
            # plt.errorbar(x, y_geometricmean, y_geometricstd, color="blue", label=m+" geometric")
            # print("2")
            # plt.errorbar(x, y_harmonicmean, y_harmonicstd, color="green", label=m+" harmonic")
            # print("3", title)
plt.xlabel("num fewshot examples")
plt.ylabel("metric")
plt.legend(loc="best")
plt.title(title)
plt.savefig(sys.argv[2])