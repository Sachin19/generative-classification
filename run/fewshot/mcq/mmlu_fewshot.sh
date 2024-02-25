task="mmlu_fewshot"
dataset=lukaemon/mmlu
data_dir=miscellaneous
split=validation
question=input
choices="A,B,C,D"
labelfield=target
label2id=None
data_files=None
jobid=$SLURM_JOB_ID
possible_labels="A,B,C,D"

# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_direct2.sh $task $dataset $data_dir $data_files $split $textfield1 $textfield2 $labelfield $jobid $label2id 
# echo "completed direct *******************"
bash run/fewshot/mcq/mcq_fewshot.sh $task $dataset $data_dir $data_files $split $question $choices $labelfield $label2id $jobid $possible_labels
# bash run/all_direct_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# bash run/all_direct_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# python plot.py final_results/main/$dataset-$data_dir/$model/results.jsonl final_results/main/$dataset-$data_dir/$model/results-plot.png 
