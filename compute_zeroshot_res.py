import sys
import numpy
import os

models = ["gpt2-xl", "EleutherAI/gpt-j-6B", "mistralai/Mistral-7B-v0.1",  "huggyllama/llama-7b", "huggyllama/llama-13b", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "tiiuae/falcon-7b"]
acc_lines = []
f1_lines = []
val_mods = []
acc_idx = int(sys.argv[2])
f1_idx = int(sys.argv[3])
res = sys.argv[4]
for model in models:
    file_name = sys.argv[1] + "/" + model + "/" + res
    # print(file_name)
    if os.path.exists(file_name):
        # val_mods.append(model)
        with open(file_name, 'r') as f:
            lines = f.readlines()
            # print(lines)
            if (len(lines) > 0):
                acc_lines.append(lines[acc_idx].split("&")[-2].strip().strip("$"))
                f1_lines.append(lines[f1_idx].split("&")[-2].strip().strip("$"))
                val_mods.append(model)
            else:
                val_mods.append(" ")
                acc_lines.append(" ")
                f1_lines.append(" ")
            # print(acc_line)
    else:
        val_mods.append(" ")
        acc_lines.append(" ")
        f1_lines.append(" ")
print("models:",",".join(val_mods))
print(",".join(acc_lines))
print(",".join(f1_lines))

    
    
