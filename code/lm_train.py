
import os

OUTPUT_DIR = './data/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

import pandas as pd
import numpy as np
import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings

warnings.filterwarnings("ignore")

import scipy as sp
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable

from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import transformers

print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

transformers.logging.set_verbosity_error()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import json
import warnings

warnings.filterwarnings('ignore')


def get_logger(filename=OUTPUT_DIR + 'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class CFG:
    apex = True
    num_workers = 0
    model = "hfl/chinese-macbert-base"
    scheduler = 'cosine'
    batch_scheduler = True
    num_cycles = 0.5
    num_warmup_steps = 1000
    epochs = 3
    last_epoch = -1
    encoder_lr = 1e-5
    decoder_lr = 1e-5
    batch_size = 8
    max_len = 200
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    seed = 2022
    n_fold = 10
    trn_fold = [0, ]
    train = True
    awp = 0
    adv_train = 0
    max_grad_norm = 1000


LOGGER = get_logger()
seed_everything(seed=CFG.seed)
train = pd.read_csv('./data/yb_train.csv', sep='\t', error_bad_lines=False)
Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train, train['label'])):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)

tokenizer = AutoTokenizer.from_pretrained(CFG.model)
CFG.tokenizer = tokenizer


def prepare_input(cfg, text):
    inputs = cfg.tokenizer(text,
                           add_special_tokens=True,
                           truncation=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df['text'].values
        self.labels = df['label'].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.text[item],
                               )
        labels = torch.tensor(self.labels[item], dtype=torch.long)
        return inputs, labels


class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        self.fc = nn.Linear(self.config.hidden_size, 2)

    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = torch.mean(outputs[0], axis=1)
        return last_hidden_states

    def loss(self, logits, labels):
        loss_fnc = nn.CrossEntropyLoss()
        loss = loss_fnc(logits, labels)
        return loss

    def forward(self, inputs, labels=None):
        feature = self.feature(inputs)
        output = self.fc(feature)
        _loss = 0
        if labels is not None:
            _loss = self.loss(output, labels)

        return output, _loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(fold, train_loader, model, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    global_step = 0
    grad_norm = 0
    tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (inputs, labels) in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds, loss = model(inputs, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if torch.cuda.device_count() > 1:
            loss = loss.mean()

        optimizer.zero_grad()

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        tk0.set_postfix(Epoch=epoch + 1, Loss=losses.avg, lr=scheduler.get_lr()[0])
    return losses.avg


def valid_fn(valid_loader, model, device):
    losses = AverageMeter()
    model.eval()
    # preds = []
    valid_true = []
    valid_pred = []
    tk0 = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for step, (inputs, labels) in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds, loss = model(inputs, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        losses.update(loss.item(), batch_size)
        batch_pred = y_preds.detach().cpu().numpy()
        for item in batch_pred:
            valid_pred.append(item.argmax(-1))
        for item in np.array(labels.cpu()):
            valid_true.append(item)
        tk0.set_postfix(Loss=losses.avg)
    print('Test set: Average loss: {:.4f}'.format(losses.avg))
    valid_true = np.array(valid_true)
    valid_pred = np.array(valid_pred)
    avg_acc = accuracy_score(valid_true, valid_pred)
    avg_f1s = f1_score(valid_true, valid_pred, average='macro')

    print('Average: Accuracy: {:.3f}%, F1Score: {:.3f}'.format(100 * avg_acc, 100 * avg_f1s))
    print(classification_report(valid_true, valid_pred))

    return avg_acc, avg_f1s, losses.avg


def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    train_dataset = TrainDataset(CFG, train_folds)
    valid_dataset = TrainDataset(CFG, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    print(len(train_loader), len(valid_loader))

    # ====================================================
    # model & optimizer
    # ====================================================
    best_score = 0.
    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, OUTPUT_DIR + 'config.pth')

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay, 'initial_lr': encoder_lr},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0, 'initial_lr': encoder_lr},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0, 'initial_lr': decoder_lr}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr,
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters)

    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps,
                num_cycles=cfg.num_cycles, last_epoch=((cfg.last_epoch + 1) / cfg.epochs) * num_train_steps
            )
        return scheduler

    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    if torch.cuda.device_count() > 1:
        print("Currently training on", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    # ====================================================
    # loop
    # ====================================================

    for epoch in range(CFG.epochs - 1 - CFG.last_epoch):

        start_time = time.time()

        avg_loss = train_fn(fold, train_loader, model, optimizer, epoch, scheduler, device)

        # eval
        avg_acc, avg_f1s, valid_loss = valid_fn(valid_loader, model, device)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f} time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {avg_f1s:.4f}')

        if best_score < avg_f1s:
            best_score = avg_f1s
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: f1: {best_score:.4f} Model')
        torch.save(model.state_dict(), OUTPUT_DIR + f"model_fold{fold}_best.bin")

    torch.cuda.empty_cache()
    gc.collect()


if CFG.train:
    for i in CFG.trn_fold:
        train_loop(train, fold=i)

############ test

test = pd.read_csv('./data/test1.csv', sep='\t')


class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df['text'].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg,
                               self.text[item],
                               )
        return inputs


def test_and_save_reault(device, test_loader, result_path):
    raw_preds = []
    test_pred = []
    for fold in CFG.trn_fold:
        current_idx = 0

        model = CustomModel(CFG, config_path=OUTPUT_DIR + 'config.pth', pretrained=True)
        model.to('cuda')
        model.load_state_dict(
            torch.load(os.path.join(OUTPUT_DIR, f"model_fold{fold}_best.bin"), map_location=torch.device('cuda')))
        model.eval()
        tk0 = tqdm(test_loader, total=len(test_loader))
        for inputs in tk0:
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                y_pred_pa_all, _ = model(inputs)
            batch_pred = (y_pred_pa_all.detach().cpu().numpy()) / CFG.n_fold
            if fold == 0:
                raw_preds.append(batch_pred)
            else:
                raw_preds[current_idx] += batch_pred
                current_idx += 1

    for preds in raw_preds:
        for item in preds:
            test_pred.append(item.argmax(-1))

    test['label'] = test_pred
    test[['id', 'label']].to_csv(result_path, index=False, sep='\t')
    return raw_preds


test_dataset = TestDataset(CFG, test)
test_loader = DataLoader(test_dataset,
                         batch_size=16,
                         shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

raw_preds = test_and_save_reault(device, test_loader, OUTPUT_DIR + 'sub.csv')