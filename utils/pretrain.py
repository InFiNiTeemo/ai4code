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