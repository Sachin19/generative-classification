# models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "EleutherAI/pythia-1.4b")
# models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("mistralai/Mistral-7B-v0.1")
# models=("gpt2-xl")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("huggyllama/llama-7b")
# models=("EleutherAI/gpt-j-6B")
# models = ("mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf")
settings=("simple")
effective_batch_size=1

task="$1"
dataset="$2"
data_dir="$3"
data_files="$4"
split="$5"
textfield1="$6"
textfield2="$7"
labelfield="$8"
label2id="$9"
jobid="${10}"
label_names="${11}"
cat_seed="${12}"
# neutr_idx="${12}"

for model in "${models[@]}"; do
    # for setting in "${settings[@]}"; do
    echo "running fewshot.py: $task  $dataset-$data_dir on $model"
    TOKENIZERS_PARALLELISM=false python fewshot.py\
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
        --batch_size 1\
        --effective_batch_size ${effective_batch_size}\
        --outputs_file "results/0923/fewshot/nli/$task/$dataset-$data_dir/$model/cat/$cat_seed/predictions.txt"\
        --results_file "results/0923/fewshot/nli/$task/$dataset-$data_dir/$model/cat/$cat_seed/results.jsonl"\
        --cc_outputs_file "results/0923/fewshot/nli/$task/$dataset-$data_dir/$model/cat/$cat_seed/cc_predictions.txt"\
        --model_dtype fp16\
        --pmi\
        --metric accuracy\
        --num_runs 10\
        --overwrite\
        --cat y\
        --cat_seed $cat_seed\
        --label_names $label_names\
        --jobid $jobid\
        --type_of_task nli_fewshot\
        
            #--bettertransformer\
    python plot_fewshot.py "results/0923/fewshot/nli/$task/$dataset-$data_dir/$model/cat/$cat_seed/results.jsonl" "results/0923/fewshot/nli/$task/$dataset-$data_dir/$model/cat/$cat_seed/results.png" "$task-$model"
    python csv_fewshot.py "results/0923/fewshot/nli/$task/$dataset-$data_dir/$model/cat/$cat_seed/results.jsonl" "results/0923/fewshot/nli/$task/$dataset-$data_dir/$model/cat/$cat_seed/results.txt" "$task-$model"
    
    # done
    # IFS='/'
    # read -ra newarr <<< "$model"
    # echo "${newarr[0]}"
    # p="models/models"
    # for r in ${newarr[@]}; do
    #     p+="--"
    #     p+=$r
    # done
    # echo $p
    # if [ -d "$p" ]; then
    #     rm -r $p
    # fi
done

if [ -d "datasets/$dataset/$data_dir" ]; then
  rm -r datasets/$dataset/$data_dir
fi
