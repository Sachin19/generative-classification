# models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "EleutherAI/pythia-1.4b")
# models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("mistralai/Mistral-7B-v0.1")
# models=("gpt2-xl")
# models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "gpt2-xl" "EleutherAI/gpt-j-6B" "tiiuae/falcon-7b")
# models=("EleutherAI/gpt-j-6B")
# models=("allenai/OLMo-7B")
# models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "mistralai/Mistral-7B-v0.1" "EleutherAI/gpt-j-6B")
# models=("EleutherAI/gpt-neox-20b" "huggyllama/llama-30b" "meta-llama/Llama-2-70b-hf")
# models=("mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-70b-hf")
models=("mistralai/Mistral-7B-Instruct-v0.1")
# models=("meta-llama/Llama-2-13b-hf")
settings=("simple")
# effective_batch_size=12

task="$1"
dataset="$2"
data_dir="$3"
data_files="$4"
split="$5"
textfield="$6"
labelfield="$7"
label2id="$8"
jobid="$9"
label_names="${10}"
effective_batch_size="${11}"

for model in "${models[@]}"; do
    for setting in "${settings[@]}"; do
        echo "running fewshot_dola_avg_jsd.py: $task $setting $dataset-$data_dir on $model"
        TOKENIZERS_PARALLELISM=false python fewshot_dola_avg_jsd.py\
            --task "$task"\
            --setting $setting\
            --dataset $dataset\
            --data_dir $data_dir\
            --split $split\
            --data_files $data_files\
            --model $model\
            --textfield1 $textfield\
            --labelfield $labelfield\
            --label2id "$label2id"\
            --batch_size 8\
            --batch_by_labelstring y\
            --effective_batch_size ${effective_batch_size}\
            --outputs_file "results/0923/fewshot/sentiment/dola/jsd-avg/$task/$dataset-$data_dir/$model/predictions.txt"\
            --results_file "results/0923/fewshot/sentiment/dola/jsd-avg/$task/$dataset-$data_dir/$model/results.jsonl"\
            --cc_outputs_file "results/0923/fewshot/sentiment/dola/jsd-avg/$task/$dataset-$data_dir/$model/cc_predictions.txt"\
            --model_dtype fp16\
            --pmi\
            --metric accuracy\
            --num_runs 10\
            --label_names $label_names\
            --jobid $jobid\
            --type_of_task nli_fewshot\
            --overwrite\

            #--debug
            
            #--bettertransformer\
        python plot_fewshot.py "results/0923/fewshot/sentiment/dola/jsd-avg/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/fewshot/sentiment/dola/jsd-avg/$task/$dataset-$data_dir/$model/results.png" "$task-$model"
        python csv_fewshot.py "results/0923/fewshot/sentiment/dola/jsd-avg/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/fewshot/sentiment/dola/jsd-avg/$task/$dataset-$data_dir/$model/results.txt" "$task-$model"

    done
done