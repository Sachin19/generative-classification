# #!/bin/bash
# #SBATCH -N 1
# #SBATCH -n 5
# #SBATCH --output=./slurm-outputs/%j.out
# #SBATCH --gres=gpu:1
# #SBATCH --mem=25g
# #SBATCH -t 0
# #SBATCH -x tir-0-9,tir-0-7,tir-0-13,tir-0-15,tir-0-17,tir-0-19,tir-0-11,tir-0-32,tir-0-36,tir-1-28,tir-1-18,tir-1-13,tir-0-3

# # set -x  # echo commands to stdout
# # set -e  # exit on error

# module load cuda-11.8
# module load gcc-7.4
# source /projects/tir1/users/sachink/data/anaconda3/bin/activate 2022

task="potato_prolific_politeness_binary_extreme#False#False"
dataset=None
data_dir=None
data_files=None
split=test
model=gpt2-large
textfield=text
labelfield=label
label2id=None

bash run/all_personal.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id

# task="potato_prolific_politeness_binary_extreme#False#True"
# dataset=None
# data_dir=None
# data_files=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# bash run/all_personal.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id

# task="potato_prolific_politeness_binary_extreme#True#False"
# dataset=None
# data_dir=None
# data_files=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# bash run/all_personal.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id

# task="potato_prolific_politeness_binary_extreme#True#True"
# dataset=None
# data_dir=None
# data_files=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# bash run/all_personal.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id