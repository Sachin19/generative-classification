#!/bin/bash
#SBATCH --job-name=cat_nli-channel-zeroshot
#SBATCH --partition=gpu-l40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu

# I use source to initialize conda into the right environment.
source activate testenv4
bash run/zeroshot/cat_nli/cat_channel/run_all.sh