# models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "EleutherAI/pythia-1.4b")
# models=("EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "gpt2-xl" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("mistralai/Mistral-7B-v0.1")
# models=("google/ul2")
# models=("gpt2-xl")
# models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("EleutherAI/gpt-j-6B")
# models=("meta-llama/Llama-2-13b-hf")
# models=("meta-llama/Llama-2-7b-hf")
# models=("meta-llama/Llama-2-7b-hf" "mistralai/Mistral-7B-v0.1" "EleutherAI/gpt-j-6B"
# models=("meta-llama/Llama-2-7b-hf" "meta-llama/Llama-2-13b-hf" "mistralai/Mistral-7B-v0.1" "EleutherAI/gpt-j-6B")
models=("EleutherAI/gpt-neox-20b" "huggyllama/llama-30b" "meta-llama/Llama-2-70b-hf")
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
hypGivenPrem="${11}"

for model in "${models[@]}"; do
    for setting in "${settings[@]}"; do
        echo "running nli_zeroshot_dola_word_by_word.py: $task $setting $dataset-$data_dir on $model, hypGivenPrem: $hypGivenPrem"
        TOKENIZERS_PARALLELISM=false python nli_zeroshot_dola_word_by_word.py\
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
            --outputs_file "results/0923/zeroshot/nli/dola/word_by_word-$task/$dataset-$data_dir/$model/predictions.txt"\
            --results_file "results/0923/zeroshot/nli/dola/word_by_word-$task/$dataset-$data_dir/$model/results.jsonl"\
            --model_dtype fp16\
            --pmi\
            --metric f1\
            --num_runs 10\
            --overwrite\
            --hypGivenPrem $hypGivenPrem\
            --jobid $jobid\
            #--debug
            
            #--bettertransformer\
        echo "plotting"
        python plot.py "results/0923/zeroshot/nli/dola/word_by_word-$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/zeroshot/nli/dola/word_by_word-$task/$dataset-$data_dir/$model/results.png" channel_
        python csv_zeroshot.py "results/0923/zeroshot/nli/dola/word_by_word-$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/zeroshot/nli/dola/word_by_word-$task/$dataset-$data_dir/$model/results.txt" channel_
    done
    # if [ ]
    # rm -r models/
done
# rm -r datasets/$dataset/$data_dir