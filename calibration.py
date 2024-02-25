import torch
from transformers import StoppingCriteriaList, MaxLengthCriteria
import numpy as np
# from fewshot import get_nll

def cc(labels, model, tokenizer, device): # labels is a list of the labels
    texts = ["N/A ", "[MASK] "]
    if tokenizer.bos_token is not None:
        texts.append(tokenizer.bos_token + " ")
    # print("***********")
    # print(labels)
    # print(texts)
    nlls = [0] * len(labels)
    for i, label in enumerate(labels):
        # print(i)
        tmp = []
        for text in texts:
            tmp_tok = tokenizer(text+label)
            # print(tmp_tok)
            inputs = {
                'attention_mask': torch.unsqueeze(torch.Tensor(tmp_tok['attention_mask']).to(torch.long).to(device), 0),
                'input_ids': torch.unsqueeze(torch.Tensor(tmp_tok['input_ids']).double().to(torch.long).to(device), 0)
            }

            label_mask = torch.zeros_like(inputs['attention_mask']).to(device)
            idx = len(tokenizer(" " + label)['input_ids'])

            label_mask[0, -idx:] = 1
            label_mask_tensor = torch.Tensor(label_mask).to(torch.long).to(device)
            tmp_val = (label_mask * inputs["input_ids"])
            # print(tokenizer.decode(tmp_val[tmp_val.ne(0)]))
            # input()

            outputs = model(**inputs)
            logits = outputs.logits
            shift_logprobs = torch.nn.functional.log_softmax(logits[..., :-1, :], dim=-1).contiguous() #everything but last in the sequence (num_class * num_labels, max_len-1, vocab_size)
            shift_target = inputs['input_ids'][..., 1:].contiguous() #everything but first in the sequence (num_class * num_labels, max_len-1)
            nll = torch.nn.functional.nll_loss(shift_logprobs.view(-1, shift_logprobs.size(-1)), shift_target.view(-1), reduction="none", ignore_index=tokenizer.pad_token_id).view(-1, shift_target.size(-1)) # to ensure they correspond and pick the correct word from the dist. reduction is to not do anything extra
            nll = nll * label_mask[..., 1:]
            deno = (shift_target.ne(tokenizer.pad_token_id).float() * label_mask[..., 1:]).sum(dim=-1) # just to account for the lengths, not too important as lengths of x are the same
            nll = nll.sum(dim=-1)/deno # actual summation
            tmp.append(nll.item())

        tmp_sum = -(torch.logsumexp(-1 * torch.Tensor(tmp), dim=0) - np.log(len(texts)))
        nlls[i] = tmp_sum
    
    return torch.Tensor(nlls)    