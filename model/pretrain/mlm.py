import numpy as np
import random
import torch
import copy


def mask_tokens_ngram(tokens, vocab_size, max_ngram=3, mask_prob=0.15):
    target = copy.deepcopy(tokens)
    # print("target:", target)
    ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, max_ngram + 1)
    pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)

    cand_indices = []
    for i, token in enumerate(tokens):
        if token in [101, 102]:
            continue
        cand_indices.append(i)

    num_to_mask = max(1, round(len(cand_indices) * mask_prob))  # 四舍五入
    random.shuffle(cand_indices)  #
    # masked_token_labels = []
    covered_indices = set()
    masked_token_num = 0
    for index in cand_indices:
        n = np.random.choice(ngrams, p=pvals)
        if masked_token_num >= num_to_mask:
            break
        if index in covered_indices:
            continue
        if index >= len(cand_indices) - (n - 1):
            continue
        # if index < len(cand_indices) - (n - 1):
        # 先选区域和长度， 然后对mask tokens, 0.8 mask， 0.1随机替换， 0.1
        for i in range(n):
            ind = index + i
            if ind in covered_indices:
                continue
            masked_token_num += 1
            covered_indices.add(ind)
            # 80% of the time, replace with [MASK]
            r = random.random()
            if r < 0.8:
                masked_token = 103
            elif r < 0.9:
                masked_token = tokens[ind]
            else:
                masked_token = random.randint(0, vocab_size-1)
            tokens[ind] = masked_token

    input_ids = tokens
    return input_ids, target



def getMaskedLabels(input_ids, tokenizer, mask_prob=0.15):
    rand = torch.rand(input_ids.shape)
    mask_arr = mask_prob
    # Preventing special tokens to get replace by the [MASK] token
    for special_token in tokenizer.special_tokens:
        token = special_token.item()
        mask_arr *= (input_ids != token)
    selection = torch.flatten(mask_arr[0].nonzero()).tolist()
    input_ids[selection] = 128000

    return input_ids
