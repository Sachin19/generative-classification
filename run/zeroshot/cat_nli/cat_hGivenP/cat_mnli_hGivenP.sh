task="mnli_hGivenP"
dataset=multi_nli
data_dir=default
split=validation_matched
textfield1=premise
textfield2=hypothesis
labelfield=label
label2id=None
data_files=None
jobid=$SLURM_JOB_ID
hypGivenPrem=y
cat_seed="$1"

# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_direct2.sh $task $dataset $data_dir $data_files $split $textfield1 $textfield2 $labelfield $jobid $label2id 
bash run/zeroshot/cat_nli/cat_nli.sh $task $dataset $data_dir $data_files $split $textfield1 $textfield2 $labelfield $label2id $jobid $hypGivenPrem $cat_seed

# bash run/all_direct_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# bash run/all_direct_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# python plot.py final_results/main/$dataset-$data_dir/$model/results.jsonl final_results/main/$dataset-$data_dir/$model/results-plot.png 