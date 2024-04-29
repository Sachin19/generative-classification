#!/bin/bash
#SBATCH --job-name=rte-zeroshot-dola
#SBATCH --partition=gpu-a40
#SBATCH --account=ark
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hsethu@uw.edu

# I use source to initialize conda into the right environment.
source activate testenv5
bash run/zeroshot/nli/hGivenP/dola_tests/jsd_avg/rte_hGivenP_jsd_avg.sh
bash run/zeroshot/nli/hGivenP/dola_tests/word_by_word/rte_hGivenP_word_by_word.sh