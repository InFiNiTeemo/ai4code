from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer, BertTokenizer

test = True
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
text = 'ldr\tw5, [x3, #0x2f74]'
if test:
    print("raw text:", text)
    # add_special_tokens=True is set by default
    text_enc = tokenizer.encode_plus(text)
    attn_mask = text_enc['attention_mask']
    text_enc = text_enc['input_ids']

    for tok in text_enc:
        print(tok, tokenizer.decode(tok))
        print(attn_mask)