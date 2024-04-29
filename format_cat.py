import sys
import os
import math

precision = 2
prefix = sys.argv[1]
suffix = sys.argv[2]
models = ["gpt2-xl", "EleutherAI/gpt-j-6B", "mistralai/Mistral-7B-v0.1", "huggyllama/llama-7b", "huggyllama/llama-13b", "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "tiiuae/falcon-7b"]
results = ["N/A"] * len(models)
valid_models = []
for i, model in enumerate(models):
    path = f"{prefix}/{model}/{suffix}"
    if os.path.isfile(path):
        valid_models.append(model)
        with open(path, 'r') as f:
            tmp = f.readlines()[1].split(",")[-1].split("+-")
            tmp = [str(round(float(t), precision)) for t in tmp]
            results[i] = "+-".join(tmp)

print(len(results))
print(valid_models)
print(",".join(results))