# models=("gpt2-xl")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
path="results/0923/zeroshot/nli"
num_runs=10
for model in "${models[@]}"; do
    bash run/compute_cat_score.sh "cb_hGivenP" "super_glue" "cb" "$model" "$path" 2 $num_runs
    bash run/compute_cat_score.sh "wnli_hGivenP" "glue" "wnli" "$model" "$path" 2 $num_runs
    bash run/compute_cat_score.sh "rte_hGivenP" "glue" "rte" "$model" "$path" 2 $num_runs
    bash run/compute_cat_score.sh "sick_hGivenP" "sick" "default" "$model" "$path" 2 $num_runs
    bash run/compute_cat_score.sh "mnli_hGivenP" "multi_nli" "default" "$model" "$path" 2 $num_runs
done