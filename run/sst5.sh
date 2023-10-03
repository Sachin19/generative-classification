task="sentiment5"
dataset="SetFit/sst5"
data_dir=None
split=test
textfield=text
labelfield=label
label2id=None
data_files=None
jobid=$SLURM_JOB_ID

# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
bash run/all_direct.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
bash run/all_channel.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

bash run/all_direct_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid True
bash run/all_channel_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid True
 
bash run/all_direct_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid True 
bash run/all_channel_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid True