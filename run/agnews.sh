task="agnews"
dataset="ag_news"
data_dir=None
data_files=None
split=test
textfield=text
labelfield=label
label2id=None
jobid=$SLURM_JOB_ID

# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
bash run/all_direct.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
bash run/all_channel.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

bash run/all_direct_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid True 
bash run/all_channel_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid True

bash run/all_direct_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid True
bash run/all_channel_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid True

# python plot.py final_results/main/$dataset-$data_dir/$model/results.jsonl final_results/main/$dataset-$data_dir/$model/results-plot.png 


# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id

# task="emotion6"
# dataset=dair-ai/emotion
# data_dir=None
# split=test
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id