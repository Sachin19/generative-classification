cat_seeds=(3)
for cat_seed in "${cat_seeds[@]}"; do
    echo $cat_seed
    # bash run/fewshot/cat_nli/cat_cb_fewshot.sh $cat_seed
    # bash run/fewshot/cat_nli/cat_wnli_fewshot.sh $cat_seed
    bash run/fewshot/cat_nli/cat_rte_fewshot.sh $cat_seed
    # bash run/fewshot/cat_nli/cat_sick_fewshot.sh $cat_seed
    # bash run/fewshot/cat_nli/cat_mnli_fewshot.sh $cat_seed
done