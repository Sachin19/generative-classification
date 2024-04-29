task="sick_fewshot"
dataset=sick
data_dir=default
split=validation
textfield1=sentence_A
textfield2=sentence_B
labelfield=label
label2id=None
data_files=None
jobid=$SLURM_JOB_ID
label_names="entailment,neutral,contradiction"
# bash run/all.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_direct2.sh $task $dataset $data_dir $data_files $split $textfield1 $textfield2 $labelfield $jobid $label2id 
# echo "completed direct *******************"
b1=(0.25)
b2=(0.5)
for i in "${!b1[@]}"; do
    bash run/fewshot/nli/dola_tests/jsd_avg/nli_fewshot_jsd_avg.sh $task $dataset $data_dir $data_files $split $textfield1 $textfield2 $labelfield $label2id $jobid $label_names 1 ${b1[i]} ${b2[i]}
done
# bash run/all_direct_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_large_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# bash run/all_direct_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid
# bash run/all_channel_extralarge_models.sh $task $dataset $data_dir $data_files $split $textfield $labelfield $label2id $jobid

# python plot.py final_results/main/$dataset-$data_dir/$model/results.jsonl final_results/main/$dataset-$data_dir/$model/results-plot.png 
