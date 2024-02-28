# models=("gpt2-xl")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")

path="results/0923/channel"
file_names=("predictions.txt")
num_runs=10
for model in "${models[@]}"; do
    for file_name in "${file_names[@]}"; do
        echo $model $file_name
        bash run/compute_cat_score.sh "results/0923/channel/cb/super_glue-cb/$model" "results/0923/channel/cb/cat/super_glue-cb/$model" 2 $num_runs $file_name
        bash run/compute_cat_score.sh "results/0923/channel/rte/glue-rte/$model" "results/0923/channel/rte/cat/glue-rte/$model" 1 $num_runs $file_name
        bash run/compute_cat_score.sh "results/0923/channel/wnli/glue-wnli/$model" "results/0923/channel/wnli/cat/glue-wnli/$model" 0 $num_runs $file_name
        bash run/compute_cat_score.sh "results/0923/channel/sick/sick-default/$model" "results/0923/channel/sick/cat/sick-default/$model" 1 $num_runs $file_name
    done
done