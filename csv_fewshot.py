import json
import numpy as np
import sys 

if len(sys.argv) >= 4:
    title=sys.argv[3]
else:
    title=""
i = 0
all_mean = []
all_std = []
for setting in ["", "cc_"]:
    ys_mean = []
    ys_std = []
    for m in ['accuracy', 'f1']:
        with open(sys.argv[1]) as f:
            y = []
            x = None
            c = 0
            for line in f:
                items = json.loads(line)
                if x is None:
                    x = items['k']
                y.append(items[setting+m])
            y_mean = np.round(np.mean(y, axis=0), 2)
            y_std = np.round(np.std(y, axis=0), 2)
            ys_mean.append(y_mean)
            ys_std.append(y_std)
    all_mean.append(ys_mean)
    all_std.append(ys_std)
            # plt.errorbar(x, y_mean, y_std, color=colors[i], label=setting+m)

f_csv = open(sys.argv[2], "w")
f_csv.write("\\begin{"+"tabular"+"}"+"{" + "|"+"|c" * (len(x)+1) + "||" + "}\n")

k_string = "type \\ num fewshot examples & "+" & ".join(map(str, x)) + " \\"+"\\"+"\n"
f_csv.write("\\hline\\hline\n")
f_csv.write(k_string)
for i, setting in enumerate(["", "cc\\_"]):
    for j, met in enumerate(['accuracy', 'f1']):
        f_csv.write("\\hline\n")
        compound = ""
        for m, s in zip(all_mean[i][j], all_std[i][j]):
            compound += "$" + str(m) + "\\pm" + str(s) + "$" + " & "
        compound = compound[:-3]
            
        string = setting+met+" & "+ compound + " \\"+"\\"+"\n"
        f_csv.write(string)
f_csv.write("\\hline\n")
f_csv.write("\\end{"+"tabular"+"}\n")
f_csv.close()
