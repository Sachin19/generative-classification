#!/bin/bash
#SBATCH --job-name=WBWMmlu
#SBATCH --partition=gpu-a100
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu

# I use source to initialize conda into the right environment.
source activate testenv5
bash run/fewshot/mcq/dola/word_by_word/mmlu_fewshot.sh