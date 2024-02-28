# models=("gpt2-xl")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
settings=("simple" "context" "instruct")

file_names=("predictions.txt" "cc_predictions.txt")
num_runs=10

for model in "${models[@]}"; do
    for setting in "${settings[@]}"; do
        for file_name in "${file_names[@]}"; do
            echo $model $file_name
            bash run/compute_cat_score.sh "results/0923/direct-$setting/cb/super_glue-cb/$model" "results/0923/direct-$setting/cb/cat/super_glue-cb/$model" 2 $num_runs $file_name
            bash run/compute_cat_score.sh "results/0923/direct-$setting/rte/glue-rte/$model" "results/0923/direct-$setting/rte/cat/glue-rte/$model" 1 $num_runs $file_name
            bash run/compute_cat_score.sh "results/0923/direct-$setting/wnli/glue-wnli/$model" "results/0923/direct-$setting/wnli/cat/glue-wnli/$model" 0 $num_runs $file_name
            bash run/compute_cat_score.sh "results/0923/direct-$setting/sick/sick-default/$model" "results/0923/direct-$setting/sick/cat/sick-default/$model" 1 $num_runs $file_name
        done
    done
done