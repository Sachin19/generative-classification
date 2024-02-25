models=("EleutherAI/gpt-j-6b" "huggyllama/llama-13b" "meta-llama/Llama-2-13b-hf" "EleutherAI/pythia-12b")
settings=("simple")
effective_batch_size=30

task="$1"
dataset="$2"
data_dir="$3"
data_files="$4"
split="$5"
textfield="$6"
labelfield="$7"
label2id="$8"
jobid="$9"
bbl=${10}
if [ -z "$bbl" ]
then
    bbl=False
fi

for model in "${models[@]}"; do
    for setting in "${settings[@]}"; do
        echo "running channel.py: $task $setting $dataset-$data_dir on $model"
        TOKENIZERS_PARALLELISM=false python channel.py\
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
            --outputs_file "results/0923/channel-$setting/$task/$dataset-$data_dir/$model/predictions.txt"\
            --results_file "results/0923/channel-$setting/$task/$dataset-$data_dir/$model/results.jsonl"\
            --model_dtype bf16\
            --pmi\
            --bettertransformer\
            --num_runs 10\
            --batch_by_label $bbl\
            --jobid $jobid

        python plot.py "results/0923/channel-$setting/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/channel-$setting/$task/$dataset-$data_dir/$model/results-plot-channel.png" channel_
    done
done
