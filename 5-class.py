# %%
import torch
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding

# %%
modelname = "gpt2-large"
tokenizer = AutoTokenizer.from_pretrained(modelname)

special=False
if special:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer.pad_token = tokenizer.eos_token    
    tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
from datasets import load_dataset

datasets = {"glue": ["cola", "sst2", "ax"], "adv_glue": ["adv_sst2"]}


import urllib.request
urllib.request.urlretrieve("https://file.io/kprHCLwTDTHr", "datasets/yoder.zip")

# datapath="tasksource/subjectivity"
# raw_dataset = load_dataset(datapath, cache_dir="datasets")
# "jigsaw_unintended_bias", data_dir="/projects/tir5/users/sachink/embed-style-transfer/data/toxicity-jigsaw/", cache_dir="datasets")

# datapath="hyperpartisan"
# raw_dataset = load_dataset("SetFit/sst5", cache_dir="datasets")
# yelp_review_full
# 

# raw_dataset = load_dataset("SetFit/sst5", cache_dir="datasets")
# print(raw_dataset['test'])

# %%
raw_dataset['validation'][:5]

# %%

# label2id = {"SUBJ": 0, "OBJ":1}
# def preprocess_function(examples):
#         x = tokenizer(examples["Sentence"], max_length=128, padding=True, truncation=True)   
#         #print(examples['Label']) 
#         x['labels'] = [label2id[label] for label in examples['Label']]
#         return x

def preprocess_function(examples):
        x = tokenizer(examples["text"], max_length=128, truncation=True)   
        return x

tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text", "label_text"])
# tokenized_dataset = tokenized_dataset.remove_columns(["Sentence", "Solved conflict", "Label"])
tokenized_dataset.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
print("datasets and tokenizer loaded")

# %%
model = AutoModelForCausalLM.from_pretrained(modelname)

if special:
    old_embeddings = model.get_input_embeddings()
    num_old_embeddings, embedding_dim = old_embeddings.weight.size()
    new_embeddings = torch.nn.Embedding(num_old_embeddings + 1, embedding_dim)

    new_embeddings.to(old_embeddings.weight.device)

    # initialize all new embeddings (in particular added tokens)
    new_embeddings.weight.data.fill_(0.)

    new_embeddings.weight.data[:num_old_embeddings, :] = old_embeddings.weight.data
    model.set_input_embeddings(new_embeddings)

    model.vocab_size = num_old_embeddings + 1
    model.config.vocab_size = model.vocab_size
    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()
print("model loaded")

# %%
model.get_output_embeddings()

# %%
old_alllabelstrings = [
    ["This statement has a negative sentiment. "],
    ["This statement has a unhappy sentiment "],
    ["This statement has a neutral sentiment. "],
    ["This statement has a happy sentiment. "],
    ["This statement has an elelated sentiment. "]
]

# old_alllabelstrings = [
#     ["This is a one star review. ", "This review receives a rating of one star. "],
#     ["This is a two star review. ", "This review receives a rating of two star. "],
#     ["This is a three star review. ", "This review receives a rating of three star. "],
#     ["This is a four star review. ", "This review receives a rating of four star. "],
#     ["This is a five star review. ", "This review receives a rating of five star. "]
# ]

alllabelstrings = []
for item in zip(*old_alllabelstrings):
    alllabelstrings.append(list(item))

alllabelstrings

# %%
# old_alllabelstrings = [
#     [  
#         "This statement is subject to interpretation. ",
#         # "The grammatical unacceptability and linguistic incorrectness of this sentence are apparent. ",
#         # "This sentence does not meet the standards of grammatical acceptability and linguistic correctness. ",
#         # "It can be stated that this sentence is grammatically unacceptable and linguistically incorrect. ",
#         # "The grammatical inaccuracy and linguistic impropriety of this sentence are evident. ",
#         # "This sentence demonstrates grammatical unacceptability and linguistic incorrectness. ",
#         # "The grammatical invalidity and linguistic inaccuracy of this sentence are apparent. ",
#         # "One can observe that this sentence is grammatically unacceptable and linguistically incorrect. ",
#         # "This sentence does not satisfy the criteria for both grammatical acceptability and linguistic correctness. ",
#         # "It is evident that this sentence is grammatically unacceptable and linguistically incorrect. "
#     ],
#     [  
#         "This statement is objectively always true. ",
#         # "The grammatical acceptability and linguistic correctness of this sentence are evident. ",
#         # "This sentence meets the standards of grammatical acceptability and linguistic correctness. ",
#         # "It can be stated that this sentence is both grammatically acceptable and linguistically correct. ",
#         # "The grammatical accuracy and linguistic propriety of this sentence are undeniable. ",
#         # "This sentence demonstrates grammatical acceptability and linguistic correctness. ",
#         # "The grammatical validity and linguistic accuracy of this sentence are apparent. ",
#         # "One can observe that this sentence is both grammatically acceptable and linguistically correct. ",
#         # "This sentence satisfies the criteria for both grammatical acceptability and linguistic correctness. ",
#         # "It is evident that this sentence is both grammatically acceptable and linguistically correct. "
#     ]
# ]

# alllabelstrings = []
# for item in zip(*old_alllabelstrings):
#     alllabelstrings.append(list(item))

# alllabelstrings

# %%
tokenizer.pad_token = tokenizer.eos_token    
tokenizer.pad_token_id = tokenizer.eos_token_id

num_labels = len(alllabelstrings[0])
num_labelstrings = len(alllabelstrings)

tokenizer.padding_side = "left"
alllabelstrings_tokenized = []
for labelstrings in alllabelstrings:
    alllabelstrings_tokenized.append(tokenizer(labelstrings, add_special_tokens=False, padding="longest", return_tensors="pt").to(device))
tokenizer.padding_side = "right"

alllabelstrings_tokenized

# %%
def process_batch(batch, alllabelstrings_tokenized, i):
    merged_labelstrings = alllabelstrings_tokenized[i]
    # print(merged_labelstrings)
    batch = {k: v.to(device) for k, v in batch.items()}
    batch_size, seq_len = batch['input_ids'].size()
    label_len = merged_labelstrings['input_ids'].size(-1)
    # print(batch_size, seq_len, label_len)
    
    expanded_batch_input_ids = batch['input_ids'].repeat_interleave(merged_labelstrings['input_ids'].size(0), dim=0) # output size = (#labels*batch_size, L)
    expanded_label_input_ids = merged_labelstrings['input_ids'].view(1, -1, label_len).expand(batch_size, -1, -1).contiguous().view(-1, label_len)
    input_ids = torch.cat([expanded_label_input_ids, expanded_batch_input_ids], dim=1)


    expanded_batch_attention_mask = batch['attention_mask'].repeat_interleave(merged_labelstrings['attention_mask'].size(0), dim=0) # output size = (#labels*batch_size, L)
    expanded_label_attention_mask = merged_labelstrings['attention_mask'].view(1, -1, label_len).expand(batch_size, -1, -1).contiguous().view(-1, label_len)
    attention_mask = torch.cat([expanded_label_attention_mask, expanded_batch_attention_mask], dim=1)
     
    labels = batch['labels']
    batch['input_ids'] = input_ids
    batch['attention_mask'] = attention_mask
    bsz = input_ids.size(0)
    if "labels" in batch:
        del batch['labels']

    # if "Label" in batch:
        # del batch['Label']
    # if "Solved conflict" in batch:
    #     del batch['Solved conflict']
    # del batch['Label']
    # del batch['Solved conflict']

    return batch, labels, batch_size, label_len

# %%
from torch.utils.data import DataLoader
data_collator.tokenizer.pad_token_id = tokenizer.eos_token_id

eval_dataloader = DataLoader(tokenized_dataset['validation'], collate_fn=data_collator, batch_size=4)

# print(tokenized_dataset['test'])
batch = next(iter(eval_dataloader))

# batch

# %%
label2id = {}
from tqdm import tqdm
import numpy as np
accurate = 0
total = 0
all_predictions = []
all_labels = []
############################################
for batch in tqdm(eval_dataloader):
# if True:
    #print(batch)
    nlls = []
    for i in range(len(alllabelstrings_tokenized)):
        new_batch, labels, batch_size, label_len = process_batch(batch, alllabelstrings_tokenized, i)
        # for i in range(new_batch['input_ids'].size(0)):
        #     print(tokenizer.decode(new_batch['input_ids'][i]))
        # print(new_batch)
        #print(new_batch['input_ids'].size())
        outputs = model(**new_batch)
        logits1 = outputs.logits
        
        shift_logprobs = torch.nn.functional.log_softmax(logits1[..., label_len-1:-1, :], dim=-1).contiguous()
        shift_target = new_batch['input_ids'][..., label_len:].contiguous()

        # prefix_logprobs =
        # prefix_target = 

        nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1))
        nll = nll.sum(dim=-1)/shift_target.ne(tokenizer.pad_token_id).float().sum(dim=-1)

        # print(nll.view(batch_size, -1))
        # nll = -torch.logsumexp(-nll.view(batch_size, num_labels, num_labelstrings[0]), dim=-1) + np.log(num_labelstrings[0])
        # nll = nll.view(batch_size, num_labels, num_labelstrings[0])#.mean(dim=-1)
        nll = nll.view(batch_size, num_labels)
        # print(nll)
        # print(nll.min(dim=1)[1].eq(batch['labels'].cuda()).int().sum().item())
        # print(nll[1])

        nlls.append(nll)
    
    nll = torch.stack(nlls, dim=2)
    nll = -torch.logsumexp(-nll, dim=-1) + np.log(num_labelstrings)

    #print(nll)
    nll = nll.min(dim=1)
    # print(nll[1])

    accurate += nll[1].eq(batch['labels'].cuda()).int().sum().item()
    total += nll[1].size(0)
    all_predictions += nll[1].tolist()
    all_labels += batch['labels'].tolist()
    #print(all_predictions)
    #print(all_labels)
    # print(accurate, "/", nll[1].size(0))


# %%
print(f"{accurate}/{total}, {accurate/total}")

from sklearn.metrics import confusion_matrix
print(confusion_matrix(all_labels, all_predictions))

print(sum(all_labels))

# %%
# # alllabelstrings = [
# #     [  
# #         "This review exhibits a negative bias.",
# #         "The inclination of this review is towards the negative side.",
# #         "There is a unfavorable slant in this review.",
# #         "The overall tone of this review is negative.",
# #         "This review shows a leaning against the subject.",
# #         "There is a negative inclination in this review.",
# #         "The overall impression of this review is pessimistic.",
# #         "This review tends to disfavor the subject being discussed.",
# #         "There is a negative inclination evident in this review.",
# #         "The general sentiment of this review is negative.",
# #     ],
# #     [
# #         "This review exhibits a positive bias.",
# #         "The inclination of this review is towards the positive side.",
# #         "There is a favorable slant in this review.",
# #         "The overall tone of this review is positive.",
# #         "This review shows a leaning in favor of the subject.",
# #         "There is a positive inclination in this review.",
# #         "The overall impression of this review is optimistic.",
# #         "This review tends to favor the subject being discussed.",
# #         "There is a positive inclination evident in this review.",
# #         "The general sentiment of this review is positive.",
# #     ]
# # ]

# # alllabelstrings = [["This review is leaning negative. "], ["This review is leaning positive. "]]
# # alllabelstrings = [["This review criticizes. "], ["This review appreciates. "]]

# alllabelstrings = [
#     [  
#         "This review exhibits a negative intent.",
#         #"The inclination of this review is towards the negative side.",
#         #"There is a unfavorable slant in this review.",
#         # "The overall tone of this review is negative.",
#         # "This review shows a leaning against the subject.",
#         "There is a negative inclination in this review.",
#         # "The overall impression of this review is pessimistic.",
#         # "This review tends to disfavor the subject being discussed.",
#         # "There is a negative inclination evident in this review.",
#         # "The general sentiment of this review is negative.",
#     ],
#     [
#         "This review exhibits a positive intent.",
#         #"The inclination of this review is towards the positive side.",
#         #"There is a favorable slant in this review.",
#         # "The overall tone of this review is positive.",
#         # "This review shows a leaning in favor of the subject.",
#         "There is a positive inclination in this review.",
#         # "The overall impression of this review is optimistic.",
#         # "This review tends to favor the subject being discussed.",
#         # "There is a positive inclination evident in this review.",
#         # "The general sentiment of this review is positive.",
#     ]
# ]

# num_labels = len(alllabelstrings)
# num_labelstrings = [len(labelstrings) for labelstrings in alllabelstrings]

# merged_labelstrings = []
# for labelstrings in alllabelstrings:
#     merged_labelstrings += labelstrings

# tokenizer.padding_side = "left"
# merged_labelstrings_tokenized = tokenizer(merged_labelstrings, add_special_tokens=False, padding=True, return_tensors="pt").to(device)
# tokenizer.padding_side = "right"

# print(len(merged_labelstrings))
# print(merged_labelstrings_tokenized['input_ids'].size())
# for key in merged_labelstrings_tokenized:
#     merged_labelstrings_tokenized[key] = merged_labelstrings_tokenized[key].view(num_labels, num_labelstrings[0], -1)

# tokenizer.pad_token = tokenizer.eos_token    
# tokenizer.pad_token_id = tokenizer.eos_token_id


