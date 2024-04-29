#!/bin/bash
#SBATCH --job-name=jsdAvgMmlu
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu

# I use source to initialize conda into the right environment.
source activate testenv5
bash run/fewshot/mcq/dola/jsd_avg/mmlu_fewshot.sh