import copy
import gc
import os
import sys
import time
import json
import random
import pickle
import optuna
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import *
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, make_dataclass
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding, AutoTokenizer
from typing import Any
# my lib
from model.model import *
from metrics import *
from utils.loss import mcrmse
from utils.util import AverageMeter, split_dataset, increment_path, timeit

# from torch.optim import AdamW # no correct bias


theme = "kaggle-ELL"
parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--seed', type=int, default=37)
parser.add_argument('--fold', type=str, default="N")
parser.add_argument('--model_name_or_path', type=str,
                    default="microsoft/deberta-v3-base")  # #'WENGSYX/Deberta-Chinese-Large')# 'hfl/chinese-macbert-base') #'microsoft/codebert-
parser.add_argument('--model_abbr', type=str, default=None)
parser.add_argument('--train_path', type=str, default=f"./data/{theme}/train.csv")
# parser.add_argument('--pair_path', type=str, default="./data/public/train/block1.bin")
parser.add_argument('--val_path', type=str, default="./data/yb_train.csv")
parser.add_argument('--test_path', type=str, default=f"./data/{theme}/test.csv")
parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--eval_times_per_epoch', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--n_folds', type=int, default=1)
parser.add_argument("--theme", type=str, default=theme)
parser.add_argument("--trained_model_path", type=str, default="./outputs")
# exp_stage
parser.set_defaults(is_experiment_stage=False)
parser.add_argument("--is_experiment_stage", action='store_true')
parser.add_argument("--n_exp_stop_fold", type=int, default=1)
# is_train
parser.set_defaults(is_train=False)
parser.add_argument('--is_train', action='store_true')
# is_test
parser.set_defaults(is_test=False)
parser.add_argument('--is_test', action='store_true')
# on_kaggle
parser.set_defaults(on_kaggle=False)
parser.add_argument('--on_kaggle', action='store_true')

args = parser.parse_args()
os.makedirs("./outputs", exist_ok=True)
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
    return model_abbr_dict.get(model_name, "unknown")


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


def remove_logger(logger):
    logger.handlers.clear()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ** settings ** #
# path
if args.model_abbr is None:
    model_abbr = get_model_abbr(args.model_name_or_path)
else:
    model_abbr = args.model_abbr
seed_everything(args.seed)


def set_args():
    os.makedirs(f"./outputs/{theme}/", exist_ok=True)
    _output_path = str(increment_path(f"./outputs/{theme}/exp"))
    # logger
    logger_path = f"{_output_path}/train_{model_abbr}"
    os.makedirs(os.path.dirname(logger_path), exist_ok=True)
    _logger = get_logger(filename=logger_path)
    return _output_path, _logger
output_path, logger = set_args()

# ** settings change when project change ** #
target_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]


# ** hyper para ** #
# ** CFG only for model config

# todo learn dataclass

@dataclass
class CFG:
    MyDataset: Any = ELLDatasetNoPadding
    # model
    MyModel: Any = ELLModelTest  # ELLModelv2
    pooler: Any = MeanPooling
    fc_dropout_rate: float = 0.2
    lr: float = 1e-5
    pooling_layers: int = 1
    is_bert_dp: bool = False
cfg = CFG()


def get_state_series():
    series = pd.Series(cfg.__dict__)
    series["epoch"] = args.epochs
    series["output_dir"] = output_path.split("/")[-1]
    stage = "exp" if args.is_experiment_stage else ("train" if args.is_train else "test")
    series["stage"] = stage
    series["total_max_len"] = args.total_max_len
    series["model_name_or_path"] = args.model_name_or_path
    return series
state_series = get_state_series()


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


def train_fold(train_df, val=None, fold=1, **kwargs):
    logger.info("\n" + "=" * 15 + ">" f"Fold {fold + 1} Training" + "<" + "=" * 15)
    # model
    model = cfg.MyModel(args.model_name_or_path, cfg, logger=logger)
    # model = cfg.MyModel(args.model_name_or_path, logger=logger, dropout_rate=cfg.dropout_rate, pooler=cfg.pooler)
    model = model.cuda()
    model = nn.DataParallel(model)

    # fold
    if val is None:
        train = train_df[train_df["kfold"] != fold]
        val = train_df[train_df["kfold"] == fold]
    else:
        train = train_df

    if args.is_experiment_stage:
        train = train.head(1000)
        val = val.head(len(val) // 2)
    # print(f"len dataset: train: {len(train)}, test: {len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # dataset
    train_ds = cfg.MyDataset(train, model_name_or_path=args.model_name_or_path,
                             md_max_len=args.md_max_len,
                             total_max_len=args.total_max_len,
                             columns=target_columns)
    val_ds = cfg.MyDataset(val, model_name_or_path=args.model_name_or_path,
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
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    train_optimization_steps = int(len(train_ds) / args.batch_size * args.epochs)
    num_warmup_steps = train_optimization_steps * 0.02
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
    #                                            num_training_steps=train_optimization_steps)  # PyTorch scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_cycles=0.5,
                                                num_training_steps=train_optimization_steps)

    best_epoch = 0
    best_val_score = -1e9
    for epoch in range(args.epochs):
        tmp_best_val_score = train_epochs(model, optimizer, scheduler, train_loader, val_loader, epoch, fold,
                                          best_val_score)
        if tmp_best_val_score > best_val_score:
            best_val_score = tmp_best_val_score
            best_epoch = epoch
        if epoch >= best_epoch + 2:
            logger.info("Early stop. ")
            break

    del model
    gc.collect()
    return best_val_score


def train_epochs(model, optimizer, scheduler, train_loader, val_loader, epoch, fold, best_val_score=0, **kwargs):
    model.train()
    # criterion = FocalLoss(class_num=2, alpha=torch.FloatTensor([0.7, 0.3]))  # num in test 0 : 1 = 0.385: 0.615
    criterion = torch.nn.L1Loss()  # torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss() # torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    eval_step = int(len(train_loader) / args.eval_times_per_epoch)

    tbar = tqdm(train_loader, file=sys.stdout)
    losses = AverageMeter()
    preds = []
    labels = []

    def eval_fn(_best_val_score, s):
        y_val, y_pred = validate(model, val_loader)
        score = -mcrmse(y_val, y_pred, len(target_columns))
        if score > _best_val_score:
            s = "[Best] " + s
            _best_val_score = score
            torch.save(model.state_dict(),
                       os.path.join(output_path, f"{model_abbr}_best_F{fold}.bin"))
        logger.info(f"{s}, Val score: {round(score, 4)}")
        return round(_best_val_score, 4)

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

        losses.update(loss.detach().cpu().item(), args.batch_size)
        preds.append(pred.detach().cpu().numpy().ravel())
        labels.append(target.detach().cpu().numpy().ravel())
        tbar.set_description(f"Epoch {epoch + 1} Loss: {np.round(losses.avg, 4)} lr: {scheduler.get_last_lr()}")

        if (idx + 1) % eval_step == 0:
            info_str = f"epoch{epoch}, fold{fold}, step {idx + 1}"
            best_val_score = eval_fn(best_val_score, info_str)

    # y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
    # val_label = [label for p1, p2, label in val_pairs]

    # logger.info(classification_report(y_val, y_pred))
    # print("val:", pd.Series(y_val).value_counts())
    # print("\npred:", pd.Series(y_pred).value_counts())
    logger.info("")
    del train_loader, val_loader
    gc.collect()
    return best_val_score


def fit_data(df):
    if df is None:
        return None
    # df['label'] = df[label_col]
    df['text'] = df['full_text']
    return df


def print_info():
    logger.info("*" * 15 + "  Info  " + "*" * 15)
    for k, v in cfg.__dict__.items():
        logger.info(f"\t{k}: {v}")


@timeit
def train_pipeline():
    print_info()
    # read data
    train = pd.read_csv(args.train_path)
    train = fit_data(train)

    # create fold
    from sklearn.model_selection import StratifiedKFold
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    # mskf = StratifiedKFold(n_splits=args.n_folds, shuffle=True) # for single label
    mskf = MultilabelStratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)  # for multiple labels
    for fold, (trn_, val_) in enumerate(mskf.split(train, train[target_columns])):
        train.loc[val_, "kfold"] = fold
    train["kfold"] = train["kfold"].astype(int)
    # logger.info(train[train["kfold"]==1].head(10))

    ##
    best_scores = []
    for f in range(args.n_folds):
        best_val_score = train_fold(train, fold=f)
        best_scores.append(round(best_val_score, 4))
        if args.is_experiment_stage and args.n_exp_stop_fold - 1 == f:
            break
    cv_score = round(float(np.mean(best_scores)), 4)
    logger.info("**** Best score in every fold: " + str(best_scores))
    logger.info("**** Best score Mean " + str(cv_score))

    def save_state():
        series = copy.deepcopy(state_series)
        series["scores"] = str(best_scores)
        series["CV_score"] = cv_score
        series["time"] = datetime.now().strftime("%m-%d %H:%M")
        output_df = pd.DataFrame([series])
        save_path = f"./outputs/{theme}/states.csv"
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            output_df = pd.concat([df, output_df], axis=0)
        logger.info("save state.")
        output_df.to_csv(save_path, index=False)

    save_state()
    return cv_score


def test_pipeline():
    print_info()
    test = pd.read_csv(args.test_path)
    test = fit_data(test)
    test[[target_columns]] = 0
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    test_ds = cfg.MyDataset(test, model_name_or_path=args.model_name_or_path,
                            md_max_len=args.md_max_len,
                            total_max_len=args.total_max_len,
                            columns=target_columns)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                             pin_memory=False, drop_last=False)

    logits = []
    for fold in range(args.n_folds):
        pth = f"{args.trained_model_path}/{theme}/{model_abbr}_best_F{fold}.bin"
        # print(pth)

        # load model
        s = time.time()
        model = cfg.MyModel(args.model_name_or_path, logger=logger).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(pth))
        logger.info(f"Load model fold {fold} cost time: {round(time.time() - s, 2)}s")

        _, pred = validate(model, test_loader)
        logits.append(pred)
        del model
        gc.collect()

    logit_preds = sum(np.array(logits)) / args.n_folds  # sum(a) equals sum(a, axis=0)
    class_preds = np.array([x for x in logit_preds]).reshape(-1, len(target_columns))

    submission = pd.read_csv(os.path.join(os.path.dirname(args.test_path), "sample_submission.csv"))
    submission[target_columns] = class_preds
    output_path = os.path.join(os.path.dirname(args.test_path), "submission.csv")
    if args.on_kaggle:
        output_path = "/kaggle/working/submission.csv"
    submission.to_csv(output_path, index=False)


def exp_pipeline():
    global output_path, logger, state_series

    def set_params(params: dict):
        for p, v in params.items():
            cfg.__setattr__(p, v)

    def set_params_and_train(params):
        set_params(params)
        val = train_pipeline()
        clear()
        return val

    def clear():
        global output_path, logger, state_series
        remove_logger(logger)
        output_path, logger = set_args()
        state_series = get_state_series()

    def greedy_optimize(meta):
        d = {}
        best_score = -1e9
        for pivot_col in meta.keys():
            for idx, val in enumerate(meta[pivot_col]):
                set_params(d)
                cfg.__setattr__(pivot_col, val)
                score = train_pipeline()
                if score > best_score:
                    best_score = score
                    d[pivot_col] = val
                clear()

    def random_walk_optimize(meta):
        # todo go back rate < .. ,
        p = random.randint(0, len(meta.keys()) - 1)
        pivot_col = list(meta.keys())[p]
        val_idx = random.randint(0, len(meta[pivot_col]) - 1)
        val = meta[pivot_col][val_idx]
        cfg.__setattr__(pivot_col, val)
        train_pipeline()
        clear()

    def optuna_optimize():
        def objective(trial):
            meta = {
                'fc_dropout_rate': trial.suggest_float("fc_dropout_rate", 0, 0.5),
                "is_bert_dp": trial.suggest_categorical("is_bert_dp", [True, False]),
                #'n_unit': trial.suggest_int("n_unit", 4, 18)
            }
            val = set_params_and_train(meta)
            return val

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=30)


    optuna_optimize()
    meta = {
        "fc_dropout_rate": [0, 0.1, 0.15, 0.2, 0.25, 0.3],
        # "lr": [2e-5, 3e-5, 4e-5, 1e-5],  # 尝试过 [1e-5, 2e-5, 3e-5, 4e-5]  1e-5明显改进
        # "pooling_layers": [3, 2, 1],
        # "is_bert_dp": [True, False] # False明显改进(再尝试一下)
    }


def main():
    if args.is_train:
        logger.info("*" * 8 + "TRAIN STAGE" + "*" * 8)
        train_pipeline()
    if args.is_test:
        logger.info("*" * 8 + "TEST STAGE" + "*" * 8)
        test_pipeline()
    if args.is_experiment_stage:
        exp_pipeline()


if __name__ == "__main__":
    main()
