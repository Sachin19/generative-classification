# models=("gpt2") # "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "EleutherAI/pythia-1.4b")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
settings=("")
effective_batch_size=50

task="$1"
dataset="$2"
data_dir="$3"
data_files="$4"
split="$5"
textfield1="$6"
textfield2="$7"
labelfield="$8"
jobid="$9" # ${10}
label2id="${10}"
cat_seed="${11}"

for model in "${models[@]}"; do
    for setting in "${settings[@]}"; do
        echo "running channel2.py: $task $setting $dataset-$data_dir on $model $split $textfield1 $textfield2 $labelfield $label2id $jobid $cat_seed"
        TOKENIZERS_PARALLELISM=false python channel2.py\
            --task "$task"\
            --dataset $dataset\
            --data_dir $data_dir\
            --split $split\
            --data_files $data_files\
            --model $model\
            --textfield1 $textfield1\
            --textfield2 $textfield2\
            --labelfield $labelfield\
            --label2id "$label2id"\
            --batch_size 8\
            --effective_batch_size ${effective_batch_size}\
            --outputs_file "results/0923/channel/$task/cat/$dataset-$data_dir/$model/$cat_seed/predictions.txt"\
            --results_file "results/0923/channel/$task/cat/$dataset-$data_dir/$model/$cat_seed/results.jsonl"\
            --model_dtype fp16\
            --pmi\
            --bettertransformer\
            --overwrite\
            --num_runs 10\
            --jobid $jobid\
            --cat_seed $cat_seed\
            --cat y\

        python plot.py "results/0923/channel/$task/cat/$dataset-$data_dir/$model/$cat_seed/results.jsonl" "results/0923/channel/$task/cat/$dataset-$data_dir/$model/$cat_seed/results-plot-channel.png" channel_
        python csv_zeroshot.py "results/0923/channel/$task/cat/$dataset-$data_dir/$model/$cat_seed/results.jsonl" "results/0923/channel/$task/cat/$dataset-$data_dir/$model/$cat_seed/results.txt" channel_
    done
done

