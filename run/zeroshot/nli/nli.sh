# models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "EleutherAI/pythia-1.4b")
# models=("EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "gpt2-xl" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("mistralai/Mistral-7B-v0.1")
# models=("google/ul2")
# models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
models=("gpt2-xl")
# models=("meta-llama/Llama-2-7b-hf")
settings=("simple")
effective_batch_size=16

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

for model in "${models[@]}"; do
    for setting in "${settings[@]}"; do
        echo "running nli.py: $task $setting $dataset-$data_dir on $model, hypGivenPrem: $hypGivenPrem"
        TOKENIZERS_PARALLELISM=false python nli.py\
            --task "$task"\
            --setting $setting\
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
            --outputs_file "results/0923/zeroshot/nli/$task/$dataset-$data_dir/$model/predictions.txt"\
            --results_file "results/0923/zeroshot/nli/$task/$dataset-$data_dir/$model/results.jsonl"\
            --model_dtype fp16\
            --pmi\
            --metric f1\
            --num_runs 10\
            --overwrite\
            --hypGivenPrem $hypGivenPrem\
            --jobid $jobid\
            #--debug
            
            #--bettertransformer\
        python plot.py "results/0923/zeroshot/nli/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/zeroshot/nli/$task/$dataset-$data_dir/$model/results.png" channel_
        python csv_zeroshot.py "results/0923/zeroshot/nli/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/zeroshot/nli/$task/$dataset-$data_dir/$model/results.txt" channel_
    done
    # if [ ]
    # rm -r models/
done
rm -r datasets/$dataset/$data_dir