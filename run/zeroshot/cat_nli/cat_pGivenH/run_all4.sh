cat_seeds=(4)
for cat_seed in "${cat_seeds[@]}"; do
    echo $cat_seed
    bash run/zeroshot/cat_nli/cat_pGivenH/cat_cb_pGivenH.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_pGivenH/cat_wnli_pGivenH.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_pGivenH/cat_rte_pGivenH.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_pGivenH/cat_sick_pGivenH.sh $cat_seed
done 

for cat_seed_ in "${cat_seeds[@]}"; do
    echo $cat_seed_
    bash run/zeroshot/cat_nli/cat_pGivenH/cat_mnli_pGivenH.sh $cat_seed_
done 