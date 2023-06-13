import sys
import torch
import os
import json
import logging

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AddedToken, DataCollatorWithPadding
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
import numpy as np
import click

logger = logging.getLogger(__name__)

def compute_metrics(p): #redefine
    # print(p.predictions)
    # print(p.predictions.size())
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # print(type(preds))
    preds = np.argmax(preds, axis=1)
    # print(preds)
    # print(type(preds))
    # print(preds == p.label_ids)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


@click.command()
@click.option('--hf-dataset-name', required=False, type=str, default=None)
@click.option('--hf-data-dir', required=False, type=str, default=None)
@click.option('--data-dir', required=False, type=str)
@click.option('--label-names', nargs=2, required=False, type=str)
@click.option('--label-ids', nargs=2, required=False, type=int)
@click.option('--train', required=False, type=str)
@click.option('--dev', required=False, type=str)
@click.option('--test', required=False, type=str)
@click.option('--filetype', required=False, type=str, help="how to reconcile different embeddings (project/freeze/..")
@click.option('--output-dir', required=True, type=str)
@click.option('--base-model', required=True, type=str)
@click.option('--block-size', required=False, default=128, type=int)
@click.option('--tokenizer-update-strategy', required=True, type=str)
@click.option('--secondary-model', default="none", required=False, type=str, help="model whose embeddings to use")
@click.option('--reconciliation-strategy', required=False, type=str, help="how to reconcile different embeddings (project/freeze/..")
def main(hf_dataset_name, hf_data_dir, data_dir, label_names, label_ids, train, dev, test, filetype, output_dir, base_model, block_size, tokenizer_update_strategy, secondary_model, reconciliation_strategy):
    assert(len(label_ids) == len(label_names))
    if hf_dataset_name is None:
        train_paths = []
        valid_paths = []
        test_paths = []
        for label in label_ids:
            train_paths.append(open(f"{data_dir}/{train}_{label}.{filetype}"))
            valid_paths.append(open(f"{data_dir}/{dev}_{label}.{filetype}"))
            test_paths.append(open(f"{data_dir}/{test}_{label}.{filetype}"))

        def create_dataset(paths, labelses):
            texts, labels = [], []
            # print(paths)
            for i, path in enumerate(paths):
                for l in path:
                    if filetype == "jsonl":
                        text = json.loads(l)["text"]
                    else:
                        text = l.strip()
                    labels.append(labelses[i])
                    texts.append(text)
                    
            print("create_dataset", len(texts), len(labels), set(labels))
            return texts, labels
        
        train_texts, train_labels = create_dataset(train_paths, label_ids)
        traindataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_texts, val_labels = create_dataset(valid_paths, label_ids)
        valdataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
        test_texts, test_labels = create_dataset(test_paths, label_ids)
        testdataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
        raw_dataset = DatasetDict({"train": traindataset, "validation": valdataset, "test": testdataset})

    else:
        raw_dataset = load_dataset(hf_dataset_name, hf_data_dir, cache_dir="hf_cache")
        if "test" not in dataset:
            logger.info("a test set is not provided, final evaluation result will be on the dev set")
            raw_dataset['test'] = raw_dataset['validation']

    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir="hf_cache")
    config = AutoConfig.from_pretrained(base_model, cache_dir="hf_cache")

    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(f"{output_dir}/checkpoint_best")
    labelid2tok = {label_id:tokenizer.encode(label_name, add_special_tokens=False)[0] for (label_id, label_name) in zip(label_ids, label_names)} 

    def preprocess_function(examples):
        return tokenizer(examples["text"], max_length=block_size, truncation=True)    

    tokenized_dataset = raw_dataset.map(preprocess_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    print("datasets and tokenizer loaded")

    
    model = AutoModelForCausalLM.from_pretrained(base_model, config=config)
    model.resize_token_embeddings(len(tokenizer))
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()

    def process_batch(batch):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        input_ids = batch['input_ids']
        seq_a = (torch.ones(input_ids.shape[0])*labelid2tok[0]).type_as(input_ids).view(-1,1)
        seq_b = (torch.ones(input_ids.shape[0])*labelid2tok[1]).type_as(input_ids).view(-1,1)
        seq_a = torch.cat((seq_a, input_ids), dim=1)[:,:-1]
        seq_b = torch.cat((seq_b, input_ids), dim=1)[:,:-1]
        input_ids = torch.cat((seq_a,seq_b),dim=0)
        
        attention_mask = batch['attention_mask']
        seq_a = (torch.ones(attention_mask.shape[0])).type_as(attention_mask).view(-1,1)
        seq_b = (torch.ones(attention_mask.shape[0])).type_as(attention_mask).view(-1,1)
        seq_a = torch.cat((seq_a, attention_mask), dim=1)[:,:-1]
        seq_b = torch.cat((seq_b, attention_mask), dim=1)[:,:-1]
        attention_mask = torch.cat((seq_a, seq_b),dim=0)

        labels = batch['labels']

        batch['input_ids'] = input_ids
        batch['attention_mask'] = attention_mask
        bsz = input_ids.size(0)
        del batch['labels']

        return batch, labels, bsz

    #### STOPPED HERE
    class GeDiTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, gedi_ratio=0, return_outputs=False):
            batch, labels, bsz = process_batch(inputs)
            # implement custom logic here
            outputs = model(**batch)
            
            logits = outputs.logits
            shift_logprobs = F.log_softmax(logits[..., :-1, :], dim=-1).contiguous()
            shift_target = batch['input_ids'][..., 1:].contiguous()
            nll = F.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(bsz, -1)
            nll = nll.sum(dim=-1) / shift_target.ne(tokenizer.pad_token_id).float().sum(dim=-1)  #TODO: SUM RATHER THAN AVG?
            
            outputs = nll.view(2, -1)
            num = outputs.gather(0, labels.view(1, -1))
            deno = torch.logsumexp(-outputs, dim=0)

            loss = gedi_ratio * num + (1-gedi_ratio) * (num + deno) #gedi_ratio = 1 ==> only generative
            loss = loss.sum()
            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
            inputs = self._prepare_input(inputs)
            with torch.no_grad():
                batch, labels, bsz = process_batch(inputs)
                # implement custom logic here
                outputs = model(**batch)

                logits = outputs.logits
                # print(logits)
                shift_logprobs = F.log_softmax(logits[..., :-1, :], dim=-1).contiguous()
                shift_target = batch['input_ids'][..., 1:].contiguous()
                nll = F.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(bsz, -1)

                # print(nll)
                nll = nll.sum(dim=-1)         
                nlogits = nll.view(2, -1)

                # print("nl", nlogits)
                num = nlogits.gather(0, labels.view(1, -1))
                deno = torch.logsumexp(-nlogits, dim=0)

                loss = num + deno
                loss = loss.sum()

            if prediction_loss_only:
                return (loss, None, None)
            
            return (loss, -nlogits.transpose(1, 0), labels)

    training_args = TrainingArguments(
        output_dir=f'{sys.argv[7]}/results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=4,  # batch size per device during training
        per_device_eval_batch_size=4,   # batch size for evaluation
        logging_dir=f'{sys.argv[7]}/logs',            # directory for storing logs
        logging_steps=100,
        evaluation_strategy="steps",
        save_total_limit=1,
        eval_steps=50,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        gradient_accumulation_steps=16,
        load_best_model_at_end=True,
    )

    print(training_args.n_gpu)
    trainer = GeDiTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=tokenized_dataset['train'],         # training dataset
        eval_dataset=tokenized_dataset['validation'],            # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    print("training finished")

    trainer.save_model(output_dir=f"{output_dir}/checkpoint_best") 
    # trainer.save_pretrained(f"{output_dir}/checkpoint_best")
    print("model saved")

    print("running evaluation now")

#     metric = load_metric("accuracy")
#     model.eval()
#     for batch in eval_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with torch.no_grad():
#             outputs = model(**batch)

#         logits = outputs.logits
#         predictions = torch.argmax(logits, dim=-1)
#         metric.add_batch(predictions=predictions, references=batch["labels"])

# metric.compute()

    metrics = trainer.evaluate(tokenized_dataset['validation'])
    print("validation", metrics)
    metrics = trainer.evaluate(tokenized_dataset['test'])
    print("test", metrics)

if __name__ == "__main__":
    main()

