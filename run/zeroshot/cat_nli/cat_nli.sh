# models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "EleutherAI/pythia-1.4b")
# models=("EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "gpt2-xl" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("mistralai/Mistral-7B-v0.1")
# models=("google/ul2")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("EleutherAI/gpt-j-6B")
# models = ("mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf")
settings=("")
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
hypGivenPrem="${11}"
cat_seed="${12}"

for model in "${models[@]}"; do
    for setting in "${settings[@]}"; do
        echo "running nli.py: $task $setting $dataset-$data_dir on $model, hypGivenPrem: $hypGivenPrem $cat_seed"
        TOKENIZERS_PARALLELISM=false python nli.py\
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
            --outputs_file "results/0923/zeroshot/nli/$task/cat/$dataset-$data_dir/$model/$cat_seed/predictions.txt"\
            --results_file "results/0923/zeroshot/nli/$task/cat/$dataset-$data_dir/$model/$cat_seed/results.jsonl"\
            --model_dtype fp16\
            --pmi\
            --metric f1\
            --num_runs 10\
            --hypGivenPrem $hypGivenPrem\
            --jobid $jobid\
            --cat_seed $cat_seed\
            --cat y
            #--debug
            
            #--bettertransformer\
        python plot.py "results/0923/zeroshot/nli/$task/cat/$dataset-$data_dir/$model/$cat_seed/results.jsonl" "results/0923/zeroshot/nli/$task/cat/$dataset-$data_dir/$model/$cat_seed/results.png" channel_
        python csv_zeroshot.py "results/0923/zeroshot/nli/$task/cat/$dataset-$data_dir/$model/$cat_seed/results.jsonl" "results/0923/zeroshot/nli/$task/cat/$dataset-$data_dir/$model/$cat_seed/results.txt" channel_
    done
    # if [ ]
    # rm -r models/
done