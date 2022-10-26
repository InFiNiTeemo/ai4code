import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model.model import *
from tqdm import tqdm
import sys
import os
from metrics import *
import torch
import argparse
from sklearn.metrics import f1_score, classification_report, accuracy_score
# from focal_loss.focal_loss import FocalLoss
from utils.focal_loss import FocalLoss
import random
import gc
import pickle
import ast
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding, AutoTokenizer
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import time
from utils.util import AverageMeter, split_dataset
# from torch.optim import AdamW # no correct bias


theme = "kaggle-ELL"
parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--seed', type=int, default=37)
parser.add_argument('--fold', type=str, default="N")
parser.add_argument('--model_name_or_path', type=str,
                    default="microsoft/deberta-v3-base")  # #'WENGSYX/Deberta-Chinese-Large')# 'hfl/chinese-macbert-base') #'microsoft/codebert-base'))
parser.add_argument('--train_path', type=str, default=f"./data/{theme}/train.csv")
# parser.add_argument('--pair_path', type=str, default="./data/public/train/block1.bin")
parser.add_argument('--val_path', type=str, default="./data/yb_train.csv")
parser.add_argument('--test_path', type=str, default=f"./data/{theme}/test.csv")
parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--n_folds', type=int, default=1)
parser.add_argument('--is_train', action='store_true')
parser.add_argument('--no_train', dest='is_train', action='store_false')
parser.set_defaults(is_train=True)

args = parser.parse_args()
if not os.path.exists("./outputs"):
    os.mkdir("./outputs")
data_dir = Path('../input/')


def get_model_abbr(model_name):
    # todo prefix match
    model_abbr_dict = {
        'microsoft/codebert-base-vocabplus': "codebertP",
        'microsoft/codebert-base': "codebert",
        "hfl/chinese-macbert-large": "macL",
        'hfl/chinese-macbert-base': "mac",
        'microsoft/deberta-base': "deb",
        "microsoft/deberta-v3-base": "deb",
        "microsoft/deberta-v3-large": "debL",
        "fnlp/bart-large-chinese": "bartL",
        "hfl/chinese-roberta-wwm-ext": "rob",
        "hfl/chinese-roberta-wwm-ext-large": "robL",
    }
    return model_abbr_dict[model_name]


def get_logger(filename='train'):
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


def get_hidden_size(s: str):
    if s.endswith("large") or s.endswith("Large"):
        return 1024
    else:
        return 768


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ** settings ** #
logger_path = f"./log/{theme}/train_{get_model_abbr(args.model_name_or_path)}"
if not os.path.exists(os.path.dirname(logger_path)):
    os.mkdir(os.path.dirname(logger_path))
if not os.path.exists(f"./outputs/{theme}/"):
    os.mkdir(f"./outputs/{theme}/")
logger = get_logger(filename=logger_path)
seed_everything(args.seed)

# ** hyper para ** #
experiment_stage = False
target_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
is_train = args.is_train  # .False #True #False
MyDataset = ELLDatasetNoPadding
MyModel = ELLModelv2


def read_data(data):
    if hasattr(data, '__getitem__'):
        data = [data['input_ids'], data['attention_mask'], data['labels']]
    # print(data[0].size()[-1])
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    """
    :return: one dimensional label & one dimensional pred
    """
    model.eval()
    tbar = tqdm(val_loader, file=sys.stdout)
    preds, labels = [], []
    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)
            with torch.cuda.amp.autocast():
                pred = model(*inputs)
            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())
    return np.concatenate(labels), np.concatenate(preds)


def get_logits(model, test_loader):
    model.eval()

    tbar = tqdm(test_loader, file=sys.stdout)
    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs).argmax(-1)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    y_pred = np.concatenate(preds)
    # test_df['label'] = y_pred
    # test_df = test_df[['id', 'label']]
    # test_df.to_csv("submission.csv", index=False, sep='\t')
    return y_pred


def get_optimizer_grouped_parameters(
        model, model_type,
        learning_rate, weight_decay,
        layerwise_learning_rate_decay
):
    no_decay = ["bias", "LayerNorm.weight"]
    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "fc" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    # initialize lrs for every layer
    num_layers = model.model.config.num_hidden_layers
    layers = [getattr(model, model_type).embeddings] + list(getattr(model, model_type).encoder.layer)
    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


def train_fold(train_df, val=None, fold=1, **kwargs):
    logger.info("\n" + "=" * 15 + ">" f"Fold {fold + 1} Training" + "<" + "=" * 15+"\n")
    # model
    model = MyModel(args.model_name_or_path, logger=logger)
    model = model.cuda()

    # fold
    if val is None:
        train = train_df[train_df["kfold"] != fold]
        val = train_df[train_df["kfold"] == fold]
    else:
        train = train_df
    # print(len(train), len(val))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # dataset
    train_ds = MyDataset(train, model_name_or_path=args.model_name_or_path,
                         md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len,
                         columns=target_columns)
    val_ds = MyDataset(val, model_name_or_path=args.model_name_or_path,
                       md_max_len=args.md_max_len,
                       total_max_len=args.total_max_len,
                       columns=target_columns)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                              collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                              pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                            pin_memory=False, drop_last=False)

    # scheduler

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    train_optimization_steps = int(len(train_ds) / args.batch_size * args.epochs)
    num_warmup_steps = train_optimization_steps * 0.02
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
    #                                            num_training_steps=train_optimization_steps)  # PyTorch scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_cycles=0.5,
                                                num_training_steps=train_optimization_steps)

    best_val_score = -1e9
    for epoch in range(args.epochs):
        best_val_score = train_epochs(model, optimizer, scheduler, train_loader, val_loader, epoch, fold,
                                      best_val_score)
    del model
    gc.collect()
    return best_val_score


def train_epochs(model, optimizer, scheduler, train_loader, val_loader, epoch, fold, best_val_score=0, **kwargs):
    model.train()
    # criterion = FocalLoss(class_num=2, alpha=torch.FloatTensor([0.7, 0.3]))  # num in test 0 : 1 = 0.385: 0.615
    criterion = torch.nn.L1Loss()  # torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss() # torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    tbar = tqdm(train_loader, file=sys.stdout)
    losses = AverageMeter()
    preds = []
    labels = []

    for idx, data in enumerate(tbar):
        inputs, target = read_data(data)

        with torch.cuda.amp.autocast():
            # print(inputs[0].size(), inputs[1].size())
            pred = model(*inputs)
            # print(pred.size(), target.size())
            loss = criterion(pred, target)
            # ce
            # output: [batch_size, nb_classes, *]
            # target [batch_size, *]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        # if idx % 10000 == 0:
        #    torch.save(model.state_dict(), f"./outputs/yb_model_{idx}.bin")

        losses.update(loss.detach().cpu().item(), args.batch_size)
        preds.append(pred.detach().cpu().numpy().ravel())
        labels.append(target.detach().cpu().numpy().ravel())
        tbar.set_description(f"Epoch {epoch + 1} Loss: {np.round(losses.avg, 4)} lr: {scheduler.get_last_lr()}")

    y_val, y_pred = validate(model, val_loader)

    # y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
    # val_label = [label for p1, p2, label in val_pairs]
    def rmse(y, yhat):
        # print(y, yhat)
        # print("avg:", np.sqrt(np.mean((yhat - y) ** 2)))
        y = y.reshape(-1, len(target_columns))
        yhat = yhat.reshape(-1, len(target_columns))
        # print(np.sqrt(np.mean((yhat - y) ** 2, axis=0)))
        return np.mean(np.sqrt(np.mean((yhat - y) ** 2, axis=0)))

    # def rmse():
    #    return 0
    score = -rmse(y_val, y_pred)
    logger.info(f"epoch{epoch} fold{fold}")
    # logger.info(classification_report(y_val, y_pred))
    # print("val:", pd.Series(y_val).value_counts())
    # print("\npred:", pd.Series(y_pred).value_counts())
    if score > best_val_score:
        logger.info("Best epoch so far.  ")
        best_val_score = score
        torch.save(model.state_dict(), f"./outputs/{theme}/{get_model_abbr(args.model_name_or_path)}_best_F{fold}.bin")
    logger.info("Validation score:" + str(score) + "\n")

    del train_loader, val_loader
    gc.collect()
    return best_val_score


def fit_data(df):
    if df is None:
        return None
    # df['label'] = df[label_col]
    df['text'] = df['full_text']
    return df


def print_info(args):
    logger.info("\n\n" + "*"*10 + "New run" + "*"*10)


def train_pipeline():
    print_info(args)
    # read data
    train = pd.read_csv(args.train_path)
    if experiment_stage:
        train = train.loc[:1000, ]
        args.epoch = 2
        args.fold = 1
    val = None
    if val is None and args.n_folds == 1:
        train, val = split_dataset(train)

    # fit data
    train, val = fit_data(train), fit_data(val)

    # fold
    # mskf = StratifiedKFold(n_splits=args.n_folds, shuffle=True) # for single label
    mskf = MultilabelStratifiedKFold(n_splits=args.n_folds, shuffle=True)  # for multiple labels
    for fold, (trn_, val_) in enumerate(mskf.split(train, train[target_columns])):
        train.loc[val_, "kfold"] = fold
    train["kfold"] = train["kfold"].astype(int)

    best_scores = []
    for f in range(args.n_folds):
        best_val_score = train_fold(train, val, fold=f)
        best_scores.append(best_val_score)
    logger.info("**** Best score in every fold: " + str(best_scores))


def test_pipeline():
    test = pd.read_csv(args.test_path)
    test = fit_data(test)
    test[[target_columns]] = 0
    test_ds = MyDataset(test, model_name_or_path=args.model_name_or_path,
                        md_max_len=args.md_max_len,
                        total_max_len=args.total_max_len,
                        columns=target_columns)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                             pin_memory=False, drop_last=False)

    logits = []
    for fold in range(args.n_folds):
        pth = f"./outputs/{theme}/{get_model_abbr(args.model_name_or_path)}_best_F{fold}.bin"

        ## load model
        s = time.time()
        model = MyModel(args.model_name_or_path, logger=logger).cuda()
        model.load_state_dict(torch.load(pth))
        logger.info(f"Load model fold {fold} cost time: {round(time.time() - s, 2)}s")

        _, pred = validate(model, test_loader)
        logits.append(pred)
        del model
        gc.collect()

    logit_preds = sum(np.array(logits)) / args.n_folds  # sum(a) equals sum(a, axis=0)
    class_preds = np.array([round(x) for x in logit_preds]).reshape(-1, len(target_columns))

    submission = pd.read_csv(os.path.join(os.path.dirname(args.test_path), "sample_submission.csv"))
    submission[target_columns] = class_preds
    submission.to_csv(os.path.join(os.path.dirname(args.test_path), "submission.csv"), index=False)


def main():
    #
    train_pipeline()
    #test_pipeline()


main()
