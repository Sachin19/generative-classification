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
# models=("mistralai/Mistral-7B-Instruct-v0.1")
# models=("meta-llama/Llama-2-13b-hf")
settings=("simple")
models=("allenai/OLMo-7B")
revisions=("step100000-tokens442B" "step300000-tokens1327B" "main" "step200000-tokens885B" "step400000-tokens1769B" "step500000-tokens2212B")
# revisions=("main")
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
dola_method="${12}"

for model in "${models[@]}"; do
    for revision in "${revisions[@]}"; do
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
                --textfield1 $textfield\
                --labelfield $labelfield\
                --label2id "$label2id"\
                --batch_size 8\
                --batch_by_labelstring y\
                --effective_batch_size ${effective_batch_size}\
                --outputs_file "results/0923/fewshot/sentiment/$task/$dataset-$data_dir/$model/$dola_method/$revision/predictions.txt"\
                --results_file "results/0923/fewshot/sentiment/$task/$dataset-$data_dir/$model/$dola_method/$revision/results.jsonl"\
                --cc_outputs_file "results/0923/fewshot/sentiment/$task/$dataset-$data_dir/$model/$dola_method/$revision/cc_predictions.txt"\
                --model_dtype fp16\
                --dola_method $dola_method\
                --pmi\
                --metric accuracy\
                --num_runs 10\
                --label_names $label_names\
                --jobid $jobid\
                --type_of_task nli_fewshot\
                --overwrite\

                #--debug
                
                #--bettertransformer\
            python plot_fewshot.py "results/0923/fewshot/sentiment/$task/$dataset-$data_dir/$model/$dola_method/$revision/results.jsonl" "results/0923/fewshot/sentiment/$task/$dataset-$data_dir/$model/$dola_method/$revision/results.png" "$task-$model"
            python csv_fewshot.py "results/0923/fewshot/sentiment/$task/$dataset-$data_dir/$model/$dola_method/$revision/results.jsonl" "results/0923/fewshot/sentiment/$task/$dataset-$data_dir/$model/$dola_method/$revision/results.txt" "$task-$model"
        done
    done
done