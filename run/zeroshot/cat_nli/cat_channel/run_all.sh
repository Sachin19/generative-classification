cat_seeds=(4)
for cat_seed in "${cat_seeds[@]}"; do
    echo $cat_seed
    bash run/zeroshot/cat_nli/cat_channel/cat_cb_channel.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_channel/cat_wnli_channel.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_channel/cat_rte_channel.sh $cat_seed
    bash run/zeroshot/cat_nli/cat_channel/cat_sick_channel.sh $cat_seed
done
for cat_seed_ in "${cat_seeds[@]}"; do
    echo $cat_seed_
    bash run/zeroshot/cat_nli/cat_channel/cat_mnli_channel.sh $cat_seed_
done