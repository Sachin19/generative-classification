task="$1"
dataset="$2"
data_dir="$3"
data_files="$4"
split="$5"
textfield="$6"
labelfield="$7"
label2id="$8"

models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "facebook/opt-2.7b" "EleutherAI/pythia-1.4b" "EleutherAI/pythia-2.8b")
# "EleutherAI/pythia-6.9b")
for model in "${models[@]}"; do
    echo "running personalized_channel.py: $task $dataset-$data_dir on $model"
    echo $model 
    python personalized_channel.py\
        --task "$task"\
        --dataset $dataset\
        --data_dir $data_dir\
        --split $split\
        --data_files $data_files\
        --model $model\
        --textfield $textfield\
        --labelfield $labelfield\
        --label2id "$label2id"\
        --batch_size 1\
        --outputs_file "final_results/personalized_channel/$task/$dataset-$data_dir/$model/predictions.txt"\
        --results_file "final_results/personalized_channel/$task/$dataset-$data_dir/$model/results.jsonl"\
        --model_dtype fp16\
        --bettertransformer\
        --num_runs 10
    python plot.py "final_results/personalized_channel/$task/$dataset-$data_dir/$model/results.jsonl" "final_results/personalized_channel/$task/$dataset-$data_dir/$model/results-plot-personalized_channel.png" channel_
done

for model in "${models[@]}"; do
    echo "running personalized_direct.py: $task $dataset-$data_dir on $model"
    echo $model 
    python personalized_direct.py\
        --task "$task"\
        --dataset $dataset\
        --data_dir $data_dir\
        --split $split\
        --data_files $data_files\
        --model $model\
        --textfield $textfield\
        --labelfield $labelfield\
        --label2id "$label2id"\
        --batch_size 1\
        --outputs_file "final_results/personalized_direct/$task/$dataset-$data_dir/$model/predictions.txt"\
        --results_file "final_results/personalized_direct/$task/$dataset-$data_dir/$model/results.jsonl"\
        --model_dtype fp16\
        --pmi\
        --bettertransformer\
        --num_runs 10

    python plot.py "final_results/personalized_direct/$task/$dataset-$data_dir/$model/results.jsonl" "final_results/personalized_direct/$task/$dataset-$data_dir/$model/results-plot-personalized_direct.png" direct_
    python plot.py "final_results/personalized_direct/$task/$dataset-$data_dir/$model/results.jsonl" "final_results/personalized_direct/$task/$dataset-$data_dir/$model/results-plot-personalized_direct++.png" direct++_
done
