cat_seeds=(1 2 3 4)
for cat_seed in "${cat_seeds[@]}"; do
    echo $cat_seed
    bash run/zeroshot/cat_nli/cat_direct/cat_cb_direct.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_direct/cat_wnli_direct.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_direct/cat_rte_direct.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_direct/cat_sick_direct.sh $cat_seed
done

for cat_seed_ in "${cat_seeds[@]}"; do
    echo $cat_seed_
    bash run/zeroshot/cat_nli/cat_direct/cat_mnli_direct.sh $cat_seed_
done