task="boolq_qGivenAC"
dataset=boolq
data_dir=default
split=validation
question=question
context=passage
labelfield=answer
label2id=None
data_files=None
jobid=$SLURM_JOB_ID
unmasked=question
possible_labels="True,False"
batch_by_labelstring=n

# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_direct2.sh $task $dataset $data_dir $data_files $split $textfield1 $textfield2 $labelfield $jobid $label2id 
# echo "completed direct *******************"
bash run/zeroshot/mcq_context/mcq_context.sh $task $dataset $data_dir $data_files $split $question $context $labelfield $label2id $jobid $unmasked $possible_labels $batch_by_labelstring

# bash run/all_direct_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# bash run/all_direct_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# python plot.py final_results/main/$dataset-$data_dir/$model/results.jsonl final_results/main/$dataset-$data_dir/$model/results-plot.png 
