import gc
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

# from https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
def feat_padding(input_ids, attention_mask, token_label, batch_length, padding_dict, padding_side):
    random_seed = None
    if padding_side == 'right':
        random_seed = 0
    elif padding_side == 'left':
        random_seed = 1
    else:
        random_seed = np.random.rand()

    mask_index = attention_mask.nonzero().reshape(-1)
    input_ids = input_ids.index_select(0, mask_index)
    token_label = token_label.index_select(0, mask_index)
    attention_mask = attention_mask.index_select(0, mask_index)
    ids_length = len(input_ids)

    if ids_length > batch_length:
        if random_seed <= 0.33:
            input_ids = input_ids[:batch_length]
            attention_mask = attention_mask[:batch_length]
            token_label = token_label[:batch_length]
        elif random_seed >= 0.66:
            input_ids = input_ids[-batch_length:]
            attention_mask = attention_mask[-batch_length:]
            token_label = token_label[-batch_length:]
        else:
            sub_length = ids_length - batch_length
            strat_idx = np.random.randint(sub_length + 1)
            input_ids = input_ids[strat_idx:strat_idx + batch_length]
            attention_mask = attention_mask[strat_idx:strat_idx + batch_length]
            token_label = token_label[strat_idx:strat_idx + batch_length]

    if ids_length < batch_length:
        add_length = batch_length - ids_length
        if random_seed <= 0.33:
            input_ids = F.pad(input_ids, (0, add_length), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (0, add_length), "constant", padding_dict['attention_mask'])
            token_label = F.pad(token_label, (0, add_length), "constant", padding_dict['input_ids'])
        elif random_seed >= 0.66:
            input_ids = F.pad(input_ids, (add_length, 0), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length, 0), "constant", padding_dict['attention_mask'])
            token_label = F.pad(token_label, (add_length, 0), "constant", padding_dict['input_ids'])
        else:
            add_length1 = np.random.randint(add_length + 1)
            add_length2 = add_length - add_length1
            input_ids = F.pad(input_ids, (add_length1, add_length2), "constant", padding_dict['input_ids'])
            attention_mask = F.pad(attention_mask, (add_length1, add_length2), "constant",
                                   padding_dict['attention_mask'])
            token_label = F.pad(token_label, (add_length1, add_length2), "constant", padding_dict['input_ids'])

    return input_ids, attention_mask, token_label


def feat_truncation(input_ids, attention_mask, batch_length, padding_side=None):
    random_seed = None
    if padding_side == 'right':
        random_seed = 0
    elif padding_side == 'left':
        random_seed = 1
    else:
        random_seed = np.random.rand()

    mask_index = attention_mask.nonzero().reshape(-1)
    input_ids = input_ids.index_select(0, mask_index)
    attention_mask = attention_mask.index_select(0, mask_index)
    ids_length = len(input_ids)

    if ids_length > batch_length:
        if random_seed <= 0.33:
            input_ids = input_ids[:batch_length]
            attention_mask = attention_mask[:batch_length]
        elif random_seed >= 0.66:
            input_ids = input_ids[-batch_length:]
            attention_mask = attention_mask[-batch_length:]
        else:
            sub_length = ids_length - batch_length
            strat_idx = np.random.randint(sub_length + 1)
            input_ids = input_ids[strat_idx:strat_idx + batch_length]
            attention_mask = attention_mask[strat_idx:strat_idx + batch_length]

    return input_ids, attention_mask
