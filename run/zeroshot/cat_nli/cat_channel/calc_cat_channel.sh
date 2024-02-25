# models=("gpt2-xl")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")

path="results/0923/channel"
file_names=("predictions.txt", "cc_predictions.txt")
num_runs=10
for model in "${models[@]}"; do
    for file_name in "${file_names[@]}"; do
        bash run/compute_cat_score.sh "cb" "super_glue" "cb" "$model" "$path" 2 $num_runs
        bash run/compute_cat_score.sh "wnli" "glue" "wnli" "$model" "$path" 0 $num_runs
        bash run/compute_cat_score.sh "rte" "glue" "rte" "$model" "$path" 2 $num_runs
        bash run/compute_cat_score.sh "sick" "sick" "default" "$model" "$path" 2 $num_runs
        bash run/compute_cat_score.sh "mnli" "multi_nli" "default" "$model" "$path" 2 $num_runs
    done
done