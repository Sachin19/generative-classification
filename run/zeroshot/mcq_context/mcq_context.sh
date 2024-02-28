# models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl" "facebook/opt-1.3b" "EleutherAI/pythia-1.4b")
# models=("EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("mistralai/Mistral-7B-v0.1")
# models=("meta-llama/Llama-2-13b-hf")
# models=("gpt2-xl")
# models=("EleutherAI/gpt-j-6B")
# models = ("mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")

settings=("simple")
effective_batch_size=1

task="$1"
dataset="$2"
data_dir="$3"
data_files="$4"
split="$5"
question="$6"
context="$7"
labelfield="$8"
label2id="$9"
jobid="${10}"
unmasked="${11}"
possible_labels="${12}"
batch_by_labelstring="${13}"



for model in "${models[@]}"; do
        echo "running mcq_context.py: $task $setting $dataset-$data_dir on $model, unmasked: $unmasked"
        TOKENIZERS_PARALLELISM=false python mcq_context.py\
            --task "$task"\
            --dataset $dataset\
            --data_dir $data_dir\
            --split $split\
            --data_files $data_files\
            --model $model\
            --question $question\
            --context $context\
            --labelfield $labelfield\
            --label2id "$label2id"\
            --batch_size 8\
            --effective_batch_size ${effective_batch_size}\
            --outputs_file "results/0923/zeroshot/mcq/$task/$dataset-$data_dir/$model/predictions.txt"\
            --results_file "results/0923/zeroshot/mcq/$task/$dataset-$data_dir/$model/results.jsonl"\
            --model_dtype fp16\
            --pmi\
            --metric accuracy\
            --num_runs 10\
            --unmasked $unmasked\
            --overwrite\
            --jobid $jobid\
            --possible_labels $possible_labels\
            #--debug
            
            #--bettertransformer\
        python plot.py "results/0923/zeroshot/mcq/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/zeroshot/mcq/$task/$dataset-$data_dir/$model/results-plot-channel.png" channel_
        python csv_zeroshot.py "results/0923/zeroshot/mcq/$task/$dataset-$data_dir/$model/results.jsonl" "results/0923/zeroshot/mcq/$task/$dataset-$data_dir/$model/results.txt" channel_
done
rm -r datasets/$dataset/$data_dir