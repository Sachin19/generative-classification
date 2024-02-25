import json
import numpy as np
import sys 
from matplotlib import pyplot as plt

# python csv_zeroshot.py json_directory csv_directory

if len(sys.argv) >= 4:
    prefix=sys.argv[3]
    print(prefix)
    if prefix == "channel++_":
        sys.exit(0)
    elif prefix == "channel_":
        prefix=""
else:
    prefix=""

y_arithmetics_means = []
y_geometrics_means = []
y_harmonics_means = []

y_arithmetics_stds = []
y_geometrics_stds = []
y_harmonics_stds = []

for m in ['f1', 'accuracy']:
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
            y_arithmetic.append(items[prefix+m+'_arithmetic'])
            y_geometric.append(items[prefix+m+'_geometric'])
            y_harmonic.append(items[prefix+m+'_harmonic'])

        y_arithmeticmean = np.round(np.mean(y_arithmetic, axis=0), 2)
        y_geometricmean = np.round(np.mean(y_geometric, axis=0), 2)
        y_harmonicmean = np.round(np.mean(y_harmonic, axis=0), 2)

        y_arithmeticstd = np.round(np.std(y_arithmetic, axis=0), 2)
        y_geometricstd = np.round(np.std(y_geometric, axis=0), 2)
        y_harmonicstd = np.round(np.std(y_harmonic, axis=0), 2)

        y_arithmetics_means.append(y_arithmeticmean)
        y_geometrics_means.append(y_geometricmean)
        y_harmonics_means.append(y_harmonicmean)

        y_arithmetics_stds.append(y_arithmeticstd)
        y_geometrics_stds.append(y_geometricstd)
        y_harmonics_stds.append(y_harmonicstd)

def compounding(means, stds): 
    compound = []
    for m, s in zip(means, stds):
        compound.append("$" + str(m) + "\\pm" + str(s) + "$")

    return compound

f_csv = open(sys.argv[2], "w")
f_csv.write("\\begin{"+"tabular"+"}"+"{" + "|"+"|c" * (len(x)+1) + "||" + "}\n")

k_string = "type \\ num label strings & "+" & ".join(map(str, x)) + " \\"+"\\" + "\n"
f_csv.write("\\hline\\hline\n")
f_csv.write(k_string)
for j, m in enumerate(['f1', 'accuracy']):
        f_csv.write("\\hline\n")
        string1 = m + "\\_arithmetic"+" & "+' & '.join(compounding(y_arithmetics_means[j], y_arithmetics_stds[j])) + " \\"+"\\" + "\n"
        string2 = m+ "\\_geometric"+" & "+' & '.join(compounding(y_geometrics_means[j], y_geometrics_stds[j])) + " \\"+"\\" + "\n"
        string3 = m+ "\\_harmonic"+" & "+' & '.join(compounding(y_harmonics_means[j], y_harmonics_stds[j])) + " \\"+"\\" + "\n"
        f_csv.write(string1)
        f_csv.write(string2)
        f_csv.write(string3)
f_csv.write("\\hline\n")
f_csv.write("\\end{"+"tabular"+"}\n")
f_csv.close()


# plt.legend(loc="best")
# plt.savefig(sys.argv[2])