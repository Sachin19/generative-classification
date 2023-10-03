
models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "EleutherAI/pythia-1.4b")
settings=("simple" "context" "instruct")
effective_batch_size=100

task="$1"
dataset="$2"
data_dir="$3"
data_files="$4"
split="$5"
textfield="$6"
labelfield="$7"
label2id="$8"
jobid="$9"

for model in "${models[@]}"; do
    for setting in "${settings[@]}"; do
        echo "running direct.py: $task $setting $dataset-$data_dir on $model"
        TOKENIZERS_PARALLELISM=false python direct.py\
            --task "$task"\
            --setting $setting\
            --dataset $dataset\
            --data_dir $data_dir\
            --split $split\
            --data_files $data_files\
            --model $model\
            --textfield $textfield\
            --labelfield $labelfield\
            --label2id "$label2id"\
            --batch_size 8\
            --effective_batch_size ${effective_batch_size}\
            --outputs_file "results/0923/direct-$setting/$task/$dataset-$data_dir/$model/predictions.txt"\
            --results_file "results/0923/direct-$setting/$task/$dataset-$data_dir/$model/results.jsonl"\
            --model_dtype fp16\
            --pmi\
            --bettertransformer\
            --num_runs 10\
            --jobid $jobid

        python plot.py "results/0923/direct-$setting/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/direct-$setting/$task/$dataset-$data_dir/$model/results-plot-direct.png" direct_
        python plot.py "results/0923/direct-$setting/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/direct-$setting/$task/$dataset-$data_dir/$model/results-plot-direct++.png" direct++_
    done
done
