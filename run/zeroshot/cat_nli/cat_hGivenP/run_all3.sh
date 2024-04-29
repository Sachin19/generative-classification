cat_seeds=(3)
for cat_seed in "${cat_seeds[@]}"; do
    echo $cat_seed
    bash run/zeroshot/cat_nli/cat_hGivenP/cat_cb_hGivenP.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_hGivenP/cat_wnli_hGivenP.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_hGivenP/cat_rte_hGivenP.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_hGivenP/cat_sick_hGivenP.sh $cat_seed
done 

for cat_seed_ in "${cat_seeds[@]}"; do
    echo $cat_seed_
    bash run/zeroshot/cat_nli/cat_hGivenP/cat_mnli_hGivenP.sh $cat_seed_
done 