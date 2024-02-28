models=("EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
# models=("mistralai/Mistral-7B-v0.1")
file_names=("predictions.txt")
cat_result_name="cat_results.txt"
save_file="cat_scores"
num_runs=4
for model in "${models[@]}"; do
    for file_name in "${file_names[@]}"; do
        echo $model $file_name
        echo "cb"
        bash run/compute_cat_score.sh "results/0923/fewshot/nli/cb_fewshot/super_glue-cb/$model" "results/0923/fewshot/nli/cb_fewshot/super_glue-cb/$model/cat" 2 $num_runs $file_name $cat_result_name $save_file
        echo "wnli"
        bash run/compute_cat_score.sh "results/0923/fewshot/nli/wnli_fewshot/glue-wnli/$model" "results/0923/fewshot/nli/wnli_fewshot/glue-wnli/$model/cat" 0 $num_runs $file_name $cat_result_name $save_file
        # echo "rte"
        # bash run/compute_cat_score.sh "results/0923/fewshot/nli/rte_fewshot/glue-rte/$model" "results/0923/fewshot/nli/rte_fewshot/glue-rte/$model/cat" 1 $num_runs $file_name $cat_result_name $save_file
        # echo "sick"
        # bash run/compute_cat_score.sh "results/0923/fewshot/nli/sick_fewshot/sick-default/$model" "results/0923/fewshot/nli/sick_fewshot/sick-default/$model/cat" 1 $num_runs $file_name $cat_result_name $save_file
    done
done