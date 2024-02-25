task="mmlu_qGivenA"
dataset=lukaemon/mmlu
data_dir=miscellaneous
split=validation
question=input
choices="A,B,C,D"
labelfield=target
label2id=None
data_files=None
jobid=$SLURM_JOB_ID
aGivenQ=n
possible_labels="A,B,C,D"
batch_by_labelstring=y
# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_direct2.sh $task $dataset $data_dir $data_files $split $textfield1 $textfield2 $labelfield $jobid $label2id 
# echo "completed direct *******************"
bash run/zeroshot/mcq/mmlu.sh $task $dataset $data_dir $data_files $split $question $choices $labelfield $label2id $jobid $aGivenQ $possible_labels $batch_by_labelstring
# bash run/all_direct_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# bash run/all_direct_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# python plot.py final_results/main/$dataset-$data_dir/$model/results.jsonl final_results/main/$dataset-$data_dir/$model/results-plot.png 
