# models=("gpt2-xl")
models=("gpt2-xl" "EleutherAI/gpt-j-6B" "mistralai/Mistral-7B-v0.1" "meta-llama/Llama-2-13b-hf" "huggyllama/llama-7b" "huggyllama/llama-13b" "meta-llama/Llama-2-7b-hf" "tiiuae/falcon-7b")
path="results/0923/zeroshot/nli"
num_runs=10
file_name="predictions.txt"
cat_result_name="cat_results.pkl"
save_file="cat_scores"

for model in "${models[@]}"; do
    echo $model $file_name
    echo "cb"
    bash run/compute_cat_score.sh "results/0923/zeroshot/nli/cb_pGivenH/super_glue-cb/$model" "results/0923/zeroshot/nli/cb_pGivenH/cat/super_glue-cb/$model" 2 $num_runs $file_name $cat_result_name $save_file
    echo "wnli"
    bash run/compute_cat_score.sh "results/0923/zeroshot/nli/wnli_pGivenH/glue-wnli/$model" "results/0923/zeroshot/nli/wnli_pGivenH/cat/glue-wnli/$model" 0 $num_runs $file_name $cat_result_name $save_file
    echo "rte"
    bash run/compute_cat_score.sh "results/0923/zeroshot/nli/rte_pGivenH/glue-rte/$model" "results/0923/zeroshot/nli/rte_pGivenH/cat/glue-rte/$model" 1 $num_runs $file_name $cat_result_name $save_file
    echo "sick"
    bash run/compute_cat_score.sh "results/0923/zeroshot/nli/sick_pGivenH/sick-default/$model" "results/0923/zeroshot/nli/sick_pGivenH/cat/sick-default/$model" 1 $num_runs $file_name $cat_result_name $save_file
    echo "mnli"
    bash run/compute_cat_score.sh "results/0923/zeroshot/nli/mnli_pGivenH/multi_nli-default/$model" "results/0923/zeroshot/nli/mnli_pGivenH/cat/multi_nli-default/$model" 2 $num_runs $file_name $cat_result_name $save_file
done