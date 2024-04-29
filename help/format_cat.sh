dir="$1"
task="$2"

echo "sick"
python format_cat.py results/0923/$dir/nli/sick_$task/sick-default cat_scores.txt
echo "mnli"
python format_cat.py results/0923/$dir/nli/mnli_$task/multi_nli-default cat_scores.txt
echo "wnli"
python format_cat.py results/0923/$dir/nli/wnli_$task/glue-wnli cat_scores.txt
echo "rte"
python format_cat.py results/0923/$dir/nli/rte_$task/glue-rte cat_scores.txt
echo "cb"
python format_cat.py results/0923/$dir/nli/cb_$task/super_glue-cb cat_scores.txt