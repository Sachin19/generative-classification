#!/bin/bash
#SBATCH --job-name=nli-hGivenP-zeroshot
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu

# I use source to initialize conda into the right environment.
source activate testenv4
bash run/zeroshot/nli/hGivenP/run_all.sh