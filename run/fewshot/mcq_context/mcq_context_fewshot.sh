# models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "EleutherAI/pythia-1.4b")
# models=("EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "gpt2-xl" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("mistralai/Mistral-7B-v0.1")
# models=("huggyllama/llama-7b")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("EleutherAI/gpt-j-6B")
# models = ("mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf")
settings=("simple")
effective_batch_size=1

task="$1"
dataset="$2"
data_dir="$3"
data_files="$4"
split="$5"
question="$6"
passage="$7"
labelfield="$8"
label2id="$9"
jobid="${10}"
possible_labels="${11}"

for model in "${models[@]}"; do
    for setting in "${settings[@]}"; do
        echo "running fewshot.py: $task $setting $dataset-$data_dir on $model"
        TOKENIZERS_PARALLELISM=false python fewshot.py\
            --task "$task"\
            --setting $setting\
            --dataset $dataset\
            --data_dir $data_dir\
            --split $split\
            --data_files $data_files\
            --model $model\
            --question $question\
            --passage $passage\
            --labelfield $labelfield\
            --label2id "$label2id"\
            --possible_labels $possible_labels\
            --batch_size 1\
            --effective_batch_size ${effective_batch_size}\
            --outputs_file "results/0923/fewshot/mcq/$task/$dataset-$data_dir/$model/predictions.txt"\
            --results_file "results/0923/fewshot/mcq/$task/$dataset-$data_dir/$model/results.jsonl"\
            --model_dtype fp16\
            --pmi\
            --metric accuracy\
            --num_runs 10\
            --aGivenQ $aGivenQ\
            --overwrite\
            --jobid $jobid\
            --type_of_task mcq_context_fewshot\
            #--debug
            
            #--bettertransformer\
        python plot_fewshot.py "results/0923/fewshot/mcq/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/fewshot/mcq/$task/$dataset-$data_dir/$model/results-plot-channel.png" "$task-$model"
    done
done
rm -r datasets/$dataset/$data_dir