task="sentiment2"
dataset=adv_glue
data_dir=adv_sst2
split=validation
textfield=text
labelfield=label
label2id=None
data_files=None
jobid=$SLURM_JOB_ID

# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
bash run/all_direct.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
bash run/all_channel.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# bash run/all_direct_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# bash run/all_direct_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# python plot.py final_results/main/$dataset-$data_dir/$model/results.jsonl final_results/main/$dataset-$data_dir/$model/results-plot.png 
