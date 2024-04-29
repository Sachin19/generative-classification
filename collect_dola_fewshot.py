import sys
import numpy as np
# from matplotlib import pyplot as plt
import pickle
import os
task = "dola/word-by-word-shift-sick_fewshot/sick-default"
# task = "sick_fewshot/sick-default"
# task = "dola/word-by-word-shift-rte_hGivenP_fewshot/glue-rte"
sub_path = ""
models = ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-70b-hf", "mistralai/Mistral-7B-v0.1", "EleutherAI/gpt-j-6B", "EleutherAI/gpt-neox-20b", "huggyllama/llama-30b"]
metrics = ["acc", "f1", "cc_acc", "cc_f1"]
results = np.full((len(models), len(metrics)), None)
relevant_lines = [4, 6, 8, 10]
for i, model in enumerate(models):
    path = f"results/0923/fewshot/nli/{task}/{model}{sub_path}/results.txt"
    print(path)
    if os.path.isfile(path):
        print("yes")
        with open(path, "r") as f:
            lines = f.readlines()
            result = []
            print(lines)
            for line in relevant_lines:
                result.append(lines[line].split(" ")[-2][1:-1])
            results[i, :] = np.array(result)
# metrics = ["acc", "f1"]#, "cc_acc", "cc_f1"]
for i, metric in enumerate(metrics):
    print(metric, "-----------")
    for j in range(len(models)):
        if results[j, i] is not None:
            print(results[j, i])
        else:
            print("N/A")

