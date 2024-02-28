#!/bin/bash
#SBATCH --job-name=mcq_context-zeroshot
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu

# I use source to initialize conda into the right environment.
source activate testenv4
bash run/zeroshot/mcq_context/boolq_aGivenQC.sh
bash run/zeroshot/mcq_context/boolq_cGivenQA.sh
bash run/zeroshot/mcq_context/boolq_qGivenAC.sh