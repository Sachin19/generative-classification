import sys
import os
import math

precision = 2
prefix = sys.argv[1]
suffix = sys.argv[2]
models = ["gpt2-xl", "EleutherAI/gpt-j-6B", "mistralai/Mistral-7B-v0.1", "huggyllama/llama-7b", "huggyllama/llama-13b", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "tiiuae/falcon-7b"]
results = ["N/A"] * len(models)
valid_models = []
acc_line = 4
for i, model in enumerate(models):
    path = f"{prefix}/{model}/{suffix}"
    # print(path)
    if os.path.isfile(path):
        valid_models.append(model)
        with open(path, 'r') as f:
            lines = f.readlines()
            # print("****", lines)
            tmp = lines[acc_line].split(" & ")[-2][1:-1]
            # tmp = [str(round(float(t), precision)) for t in tmp]
            results[i] = tmp

print(len(results))
print(valid_models)
print(",".join(results))