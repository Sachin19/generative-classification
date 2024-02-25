task="$1"
dataset="$2"
data_dir="$3"
model="$4"
path="$5" # expect to be .../ (don't include $task/cat/$dataset_$data-dir/$model/ here), cat would be with a task/ in between and seed at end
# result="$6"
pred_file_name="$6"
neutral_idx="$7"
num_runs="$8"
seeds=(1 2 3 4)
cat_score_file="cat_scores.txt"

org_dir="$path/$task/$dataset-$data_dir/$model"
cat_dir="$path/$task/cat/$dataset-$data_dir/$model"
res_dir="$path/$task/cat/$dataset-$data_dir/$model"

for seed in "${seeds[@]}"; do
    cat_dir_seed="$cat_dir/$seed"
    res_dir_seed="$res_dir/$seed/$cat_score_file"
    echo $org_dir
    echo $cat_dir_seed
    echo $res_dir_seed
    echo " "
    python calc_cat.py "$org_dir" "$cat_dir_seed" "$res_dir_seed" $neutral_idx $num_runs
done

save_dir="$path/$task/cat/$dataset-$data_dir/$model/overall_cat.txt"
python compile_cat.py "$cat_dir" 4 "$file_name" "$save_dir"