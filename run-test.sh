export HF_HOME=/scratch/sachink/huggingface
export HF_DATASETS_CACHE=/scratch/sachink/huggingface
module load cuda-11.8

# model=/projects/tir6/general/sachink/personalized-LM/2023/llama/hf_model-7B
# model=google/flan-t5-large
# model=google/flan-t5-xl
# model=google/flan-t5-xxl
# model=openlm-research/open_llama_3b
model=EleutherAI/pythia-1.4b
# model=EleutherAI/pythia-2.8b-deduped
# model=EleutherAI/pythia-6.9b
# model=EleutherAI/pythia-12b
# model=meta-llama/Llama-2-7b-hf
# model=meta-llama/Llama-2-13b-hf
# model=meta-llama/Llama-2-70b-hf
# model=huggyllama/llama-7b
# model=/projects/tir6/general/sachink/personalized-LM/2023/llama/hf_model-7B
# model=gpt2-large
# model="EleutherAI/gpt-j-6b"

# task="sentiment2#2#0"
# dataset=glue
# data_dir=sst2
# split=validation
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="tweet_eval-sentiment"
# dataset=tweet_eval
# data_dir=None
# split=test
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="tweet_eval-hate"
# dataset=tweet_eval
# data_dir=None
# split=test
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="cb"
# dataset=super_glue
# data_dir=cb
# split=validation
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="sentiment2-mr"
# dataset=rotten_tomatoes
# data_dir=None
# split=validation
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

task="sentiment2-yelp"
dataset=yelp_polarity
data_dir=None
split=test
textfield=text
labelfield=label
label2id=None
data_files=None

# task="sentiment2-cr"
# dataset="SetFit/CR"
# data_dir=None
# split=test
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="dbpedia"
# dataset=dbpedia_14
# data_dir=None
# split=test
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="poemsentiment2"
# dataset=poem_sentiment
# data_dir=None
# split=validation
# textfield=verse_text
# labelfield=label
# label2id=None
# data_files=None

# task="finance_sentiment3"
# dataset=financial_phrasebank
# data_dir=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="sentiment5"
# dataset="SetFit/sst5"
# data_dir=None
# split=test
# model=gpt2-xl
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="sentiment5"
# dataset="yelp_review_full"
# data_dir=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="subjectivity"
# dataset="tasksource/subjectivity"
# data_dir=None
# split=test
# model=gpt2-large
# textfield=Sentence
# labelfield=Label
# label2id="{'SUBJ': 0, 'OBJ': 1}"
# data_files=None

# task="toxicity"
# dataset="json"
# data_dir=None
# data_files="/projects/tir5/users/sachink/embed-style-transfer/data/toxicity-jigsaw/test.jsonl"
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# task="agnews"
# dataset="ag_news"
# data_dir=None
# data_files=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# task="dbpedia"
# dataset="dbpedia_14"
# data_dir=None
# data_files=None
# split=test
# textfield=text
# labelfield=label
# label2id=None

# task="hate_speech18"
# dataset="hate_speech18"
# data_dir=None
# data_files=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# task="ethos-sexual_orientation"
# dataset="ethos-sexual_orientation"
# data_dir=None 
# data_files=None
# split=test
# # model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# task="ethos-national_origin"
# dataset="ethos-national_origin"
# data_dir=None
# data_files=None
# split=test
# # model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# task="ethos-religion"
# dataset="ethos-religion"
# data_dir=None
# data_files=None
# split=test    
# # model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# task="ethos-race"
# dataset="ethos-race"
# data_dir=None
# data_files=None
# split=test
# # model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# task="potato_prolific_politeness_binary_extreme#False#True"
# # task="hate_demographic#white people"
# dataset="None"
# data_dir=None
# data_files=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# task="emotion6"
# dataset=dair-ai/emotion
# data_dir=None
# split=test
# # model=openlm-research/open_llama_3b
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="emo"
# dataset=emo
# data_dir=None
# split=test
# # model=openlm-research/open_llama_3b
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

# task="dbpedia"
# dataset=dbpedia_14
# data_dir=None
# split=test
# # model=openlm-research/open_llama_3b
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None
# data_files=Nonep

# task="talk_up#True#False"
# dataset=None
# data_dir=None
# data_files=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None

# task="social_bias_frames#lewd"
# dataset=None
# data_dir=None
# data_files=None
# split=test
# model=gpt2-large
# textfield=text
# labelfield=label
# label2id=None
# task="sentiment5"
# dataset="SetFit/sst5"
# data_dir=None
# split=test
# textfield=text
# labelfield=label
# label2id=None
# data_files=None

jobid=$SLURM_JOB_ID
# batch_size=4

TOKENIZERS_PARALLELISM=false python $1.py\
    --setting $2\
    --task "$task"\
    --dataset $dataset\
    --data_dir $data_dir\
    --split $split\
    --data_files $data_files\
    --model $model\
    --textfield $textfield\
    --labelfield $labelfield\
    --label2id "$label2id"\
    --batch_size 8\
    --effective_batch_size 200\
    --outputs_file "results/$1/$task-$data_dir/debugging/$model/predictions.txt"\
    --results_file "results/$1/$task-$data_dir/debugging/$model/results.jsonl"\
    --model_dtype bf16\
    --pmi\
    --metric $3\
    --num_runs 10\
    --jobid $SLURM_JOB_ID\
    --batch_by_label\
    --overwrite\
    --bettertransformer\
    # --debug\
    # --batch_by_labelstring\  
    

python plot.py "results/$1/$task-$data_dir/debugging/$model/results.jsonl" "results/$1/$task-$data_dir/debugging/$model/results-plot-$1.png" ${1}_
python plot.py "results/$1/$task-$data_dir/debugging/$model/results.jsonl" "results/$1/$task-$data_dir/debugging/$model/results-plot-$1++.png" ${1}++_
