import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import AdamW
from tqdm import tqdm
import os

class CFG:
    seed = 42
    model_name = 'microsoft/deberta-v3-base'
    epochs = 5
    batch_size = 4
    lr = 1e-6
    weight_decay = 1e-6
    max_len = 512
    mask_prob = 0.15  # perc of tokens to convert to mask
    n_accumulate = 4
    use_2021 = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=CFG.seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed()

if CFG.use_2021:
    competition_path = "../input/feedback-prize-2021/"
    df = pd.read_csv('../input/feedback-pseudo-labelling-full-2021-dataset/train_2021_preds.csv');
    df = df[df['in_2022'] == False]
else:
    competition_path = "../input/feedback-prize-effectiveness/"
    df = pd.read_csv(competition_path + 'train.csv')


def fetch_essay_texts(df, train=True):
    if train:
        base_path = competition_path + 'train/'
    else:
        base_path = competition_path + 'test/'

    essay_texts = {}
    for filename in os.listdir(base_path):
        with open(base_path + filename) as f:
            text = f.readlines()
            full_text = ' '.join([x for x in text])
            essay_text = ' '.join([x for x in full_text.split()])
        essay_texts[filename[:-4]] = essay_text
    df['essay_text'] = [essay_texts[essay_id] for essay_id in df['essay_id'].values]
    return df


fetch_essay_texts(df)

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
model = AutoModelWithLMHead.from_pretrained(CFG.model_name)

special_tokens = tokenizer.encode_plus('[CLS] [SEP] [MASK] [PAD]',
                                        add_special_tokens = False,
                                        return_tensors='pt')
special_tokens = torch.flatten(special_tokens["input_ids"])
print(special_tokens)


def getMaskedLabels(input_ids):
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < CFG.mask_prob)
    # Preventing special tokens to get replace by the [MASK] token
    for special_token in special_tokens:
        token = special_token.item()
        mask_arr *= (input_ids != token)
    selection = torch.flatten(mask_arr[0].nonzero()).tolist()
    input_ids[selection] = 128000

    return input_ids

def getMaskedLabelsNgram(input_ids):
    ...

class simdatasets(object):

    def __init__(self, query, candidate,
                 label, tokenizer,
                 max_seq_len, pretrain=False):

        self.query = query
        self.candidate = candidate
        self.label = label
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pretrain = pretrain
        self.MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

    def __len__(self):
        return len(self.query)

    def __getitem__(self, item):

        if self.pretrain:

            tokenize_result = self.tokenizer.encode_plus(self.query[item],
                                                         self.candidate[item],
                                                         max_length=self.max_seq_len,
                                                         truncation=True,
                                                         truncation_strategy='longest_first', )

            if self.label[item] != -1:
                tokenize_result['input_ids'] = tokenize_result['input_ids'] + [self.label[item] + 1] + [102]

                tokenize_result['attention_mask'] = tokenize_result['attention_mask'] + [1] + [1]
                tokenize_result['token_type_ids'] = tokenize_result['token_type_ids'] + [0] + [0]

            input_ids, labels = self._mask_tokens(tokenize_result['input_ids'])

            return {
                'input_ids': input_ids,
                'attention_mask': tokenize_result['attention_mask'],
                'token_type_ids': tokenize_result['token_type_ids'],
                'label': labels}



        else:
            tokenize_result = self.tokenizer.encode_plus(self.query[item],
                                                         self.candidate[item],
                                                         max_length=self.max_seq_len,
                                                         truncation=True,
                                                         truncation_strategy='longest_first', )

            if self.label[item] is not None:
                return {
                    'input_ids': tokenize_result["input_ids"],
                    'attention_mask': tokenize_result["attention_mask"],
                    'token_type_ids': tokenize_result["token_type_ids"],
                    'label': self.label[item]
                }
            return {
                'input_ids': tokenize_result["input_ids"],
                'attention_mask': tokenize_result["attention_mask"],
                'token_type_ids': tokenize_result["token_type_ids"],
            }

    def _mask_tokens(self, inputs):

        def single_mask_tokens(tokens, max_ngram=3):
            ngrams = np.arange(1, max_ngram + 1, dtype=np.int64)
            pvals = 1. / np.arange(1, max_ngram + 1)
            pvals /= pvals.sum(keepdims=True)  # p(n) = 1/n / sigma(1/k)
            cand_indices = []
            for (i, token) in enumerate(tokens):
                if token == 101 or token == 102:
                    continue
                cand_indices.append(i)

            num_to_mask = max(1, int(round(len(tokens) * 0.15)))  # 四舍五入
            random.shuffle(cand_indices)  #
            masked_token_labels = []
            covered_indices = set()

            for index in cand_indices:
                n = np.random.choice(ngrams, p=pvals)
                if len(masked_token_labels) >= num_to_mask:
                    break
                if index in covered_indices:
                    continue
                if index < len(cand_indices) - (n - 1):  # 先选区域和长度， 然后对0.8 mask， 0.1随机替换， 0.1
                    for i in range(n):
                        ind = index + i
                        if ind in covered_indices:
                            continue
                        covered_indices.add(ind)
                        # 80% of the time, replace with [MASK]
                        if random.random() < 0.8:
                            masked_token = 103
                        else:
                            # 10% of the time, keep original
                            if random.random() < 0.5:
                                masked_token = tokens[ind]
                            # 10% of the time, replace with random word
                            else:
                                masked_token = random.choice(range(0, self.tokenizer.vocab_size))
                        masked_token_labels.append(self.MaskedLmInstance(index=ind, label=tokens[ind]))
                        tokens[ind] = masked_token

            masked_token_labels = sorted(masked_token_labels, key=lambda x: x.index)

            target = len(tokens) * [-100]
            for p in masked_token_labels:
                target[p.index] = p.label

            return tokens, target

        a_mask_tokens, ta = single_mask_tokens(inputs)

        return a_mask_tokens, ta


class MLMDataset:
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        tokenized_data = self.tokenizer.encode_plus(
            text,
            max_length=CFG.max_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = torch.flatten(tokenized_data.input_ids)
        attention_mask = torch.flatten(tokenized_data.attention_mask)
        labels = getMaskedLabels(input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


essay_data = df["essay_text"].unique()
dataset = MLMDataset(essay_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True)

len(df), len(essay_data)

optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)


def train_loop(model, device):
    model.train()
    batch_losses = []
    loop = tqdm(dataloader, leave=True)
    for batch_num, batch in enumerate(loop):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        batch_loss = loss / CFG.n_accumulate
        batch_losses.append(batch_loss.item())

        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=batch_loss.item())
        batch_loss.backward()

        if batch_num % CFG.n_accumulate == 0 or batch_num == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            model.zero_grad()

    return np.mean(batch_losses)

import wandb
# from kaggle_secrets import UserSecretsClient

# user_secrets = UserSecretsClient()
# secret_value_0 = user_secrets.get_secret("WANDB")
#wandb.login(key=secret_value_0)
#wandb.init(project='feedback-prize-effectiveness', name='mlm-aurora-large')


device = CFG.device
model.to(device)
history = []
best_loss = np.inf
prev_loss = np.inf
model.gradient_checkpointing_enable()
print(f"Gradient Checkpointing: {model.is_gradient_checkpointing}")

for epoch in range(CFG.epochs):
    loss = train_loop(model, device)
    history.append(loss)
    print(f"Loss: {loss}")
    if loss < best_loss:
        print("New Best Loss {:.4f} -> {:.4f}, Saving Model".format(prev_loss, loss))
        # torch.save(model.state_dict(), "./deberta_mlm.pt")
        model.save_pretrained('./')
        best_loss = loss
    prev_loss = loss