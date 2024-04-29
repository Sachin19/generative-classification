#!/bin/bash
#SBATCH --job-name=wbw-sick-dola
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=13:20:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu

# I use source to initialize conda into the right environment.
source activate testenv5
bash run/fewshot/nli/dola_tests/word_by_word/sick_fewshot_word_by_word.sh