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
from torch.optim import AdamW  # no correct bias
from dataclasses import dataclass, make_dataclass
from sklearn.metrics import f1_score, classification_report, accuracy_score
from transformers import get_cosine_schedule_with_warmup, DataCollatorWithPadding, AutoTokenizer
from typing import Any
# my lib
from metrics import *
from model.model import *
from model.lr import get_optimizer_grouped_parameters
from model.adversial.AWP import AWP
from model.adversial.FGM import FGM
from utils.time_func import timeSince, timeit
from utils.loss import mcrmse
from utils.util import AverageMeter, split_dataset, increment_path

theme = "kaggle-ELL"
parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--model_name_or_path', type=str,
                    default="microsoft/deberta-v3-base")  # #'WENGSYX/Deberta-Chinese-Large')# 'hfl/chinese-macbert-base') #'microsoft/codebert-
parser.add_argument('--model_abbr', type=str, default=None)
parser.add_argument('--eval_times_per_epoch', type=int, default=1)
parser.add_argument('--seed', type=int, default=37)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--n_folds', type=int, default=1)
parser.add_argument("--theme", type=str, default=theme)
# cfg
parser.add_argument("--cfg_path", type=str, default=None)
parser.set_defaults(parallel=False)
parser.add_argument("--parallel", action='store_true')
# exp_stage
parser.set_defaults(is_experiment_stage=False)
parser.add_argument("--is_experiment_stage", action='store_true')
parser.add_argument("--n_exp_stop_fold", type=int, default=None)
# is_train
parser.set_defaults(is_train=False)
parser.add_argument('--is_train', action='store_true')
# is_test and is_oof
parser.set_defaults(is_oof=False)
parser.add_argument('--is_oof', action='store_true')
parser.set_defaults(is_test=False)
parser.add_argument('--is_test', action='store_true')
parser.add_argument("--test_model_path", type=str, default=None)
# on_kaggle
parser.set_defaults(on_kaggle=False)
parser.add_argument('--on_kaggle', action='store_true')
# output path
parser.add_argument('--output_base_dir', type=str, default=f"./output")
parser.add_argument('--train_path', type=str, default=f"./data/{theme}/train.csv")
parser.add_argument('--val_path', type=str, default="./data/yb_train.csv")
parser.add_argument('--test_path', type=str, default=f"./data/{theme}/test.csv")


args = parser.parse_args()
os.makedirs("./outputs", exist_ok=True)


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


def remove_handler(logger):
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

theme_path = f"./outputs/{theme}/"
def set_args():
    os.makedirs(theme_path, exist_ok=True)
    _output_path = str(increment_path(f"{theme_path}/exp"))
    # logger
    logger_path = f"{_output_path}/train_{model_abbr}"
    os.makedirs(os.path.dirname(logger_path), exist_ok=True)
    _logger = get_logger(filename=logger_path)
    return _output_path, _logger
output_path, logger = set_args()
oof_train_path = os.path.join(theme_path, f"oof_{args.seed}.pkl")
oof_output_path = os.path.join(output_path, f"oof_df_{args.seed}.pkl")


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
    backbone: Any = args.model_name_or_path  # just to save config
    # setting
    oof = False
    apex = True
    parallel: bool = args.on_kaggle and args.parallel
    print_freq = 20
    is_early_stop: bool = False
    gradient_checkpointing: bool = True
    accumulation_steps: int = 1
    batch_size: int = 8  # can significantly affect performance
    total_max_len: int = 512
    quick_exp = False
    max_grad_norm = 1000
    # adversial
    attacker: Any = None  # FGM about twice the time
    adversial_kwargs: dict = None  # mutable default dict() is not allowed
    # optimizer
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-6
    # lr
    lr: float = 1e-5 # also encoder_lr
    is_llrd = True
    llrd_kwargs = {
        "new_module_lr": lr
    }
    # para
    pooler: Any = MeanPooling
    fc_dropout_rate: float = 0.3 # 当batch=2有较大影响， 当batch=8几乎没影响
    pooling_layers: int = 1
    is_bert_dp: bool = False
    reinit_layer_num: int = 1
    # memory
    oof_score = 1e9

def save_cfg(pth=None):
    if pth is None:
        pth = output_path
    d = cfg.__dict__
    cfg_pth = os.path.join(pth, "cfg.bin")
    pickle.dump(d, open(cfg_pth, "wb+"))
def load_cfg(cfg_pth, _cfg=None):
    if _cfg is None:
        _cfg = cfg
    logger.info("* Load config from: " + cfg_pth)
    d = pickle.load(open(cfg_pth, "rb"))
    for k, v in d.items():
        cfg.__setattr__(k, v)
def show_cfg(cfg_pth):
    tmp_cfg = CFG()
    load_cfg(cfg_pth, tmp_cfg)
    print_info(tmp_cfg)

cfg = CFG()
if args.cfg_path is not None:
    load_cfg(args.cfg_path)


def oof():
    oof_df = pd.read_pickle(oof_output_path)
    label = oof_df[target_columns].values
    y_pred = oof_df[[f"pred_{c}" for c in target_columns]].values
    score = mcrmse(label, y_pred, len(target_columns))
    logger.info(f'Score: {score:<.4f} ')
    cfg.oof_score = score


def get_state_series():
    series = pd.Series(cfg.__dict__)
    series["epoch"] = args.epochs
    series["output_dir"] = output_path.split("/")[-1]
    stage = "exp" if args.is_experiment_stage else ("train" if args.is_train else "test")
    series["stage"] = stage
    # series["total_max_len"] = cfg.total_max_len
    series["model_name_or_path"] = args.model_name_or_path
    return series



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
            preds.append(pred.detach().cpu().numpy().ravel())  # ravel() -> 1d-array
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

@timeit(logger)
def train_fold(train_df, val=None, fold=1, **kwargs):
    logger.info("\n" + "=" * 15 + ">" f"Fold {fold + 1} Training" + "<" + "=" * 15)
    save_cfg()
    # model
    model = cfg.MyModel(args.model_name_or_path, cfg, logger=logger).cuda()
    # model = cfg.MyModel(args.model_name_or_path, logger=logger, dropout_rate=cfg.dropout_rate, pooler=cfg.pooler).cuda()
    if cfg.parallel:
        model = nn.DataParallel(model)

    # fold
    if val is None:
        train = train_df[train_df["kfold"] != fold]
        val = train_df[train_df["kfold"] == fold]
    else:
        train = train_df

    if args.is_experiment_stage and cfg.quick_exp:
        train = train.head(1000)
        val = val.head(len(val) // 2)
    # print(f"len dataset: train: {len(train)}, test: {len(val)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # dataset
    train_ds = cfg.MyDataset(train, model_name_or_path=args.model_name_or_path,
                             total_max_len=cfg.total_max_len,
                             columns=target_columns)
    val_ds = cfg.MyDataset(val, model_name_or_path=args.model_name_or_path,
                           total_max_len=cfg.total_max_len,
                           columns=target_columns)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=args.n_workers,
                              collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                              pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=args.n_workers,
                            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                            pin_memory=False, drop_last=False)

    # scheduler

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    if cfg.is_llrd:
        optimizer_grouped_parameters = \
            get_optimizer_grouped_parameters(model, cfg.MyModel, encoder_lr=cfg.lr, is_parallel=cfg.parallel, **cfg.llrd_kwargs)
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr,
                      correct_bias=False, eps=cfg.eps,
                      betas=cfg.betas)  # To reproduce BertAdam specific behavior set correct_bias=False
    train_optimization_steps = int(len(train_ds) / cfg.batch_size * args.epochs)
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
        if cfg.is_early_stop and epoch >= best_epoch + 2:
            logger.info("Early stop. ")
            break

    del model, train_ds, val_ds
    gc.collect()
    return best_val_score


def train_epochs(model, optimizer, scheduler, train_loader, val_loader, epoch, fold, best_val_score=0, **kwargs):
    model.train()
    # criterion = FocalLoss(class_num=2, alpha=torch.FloatTensor([0.7, 0.3]))  # num in test 0 : 1 = 0.385: 0.615
    # criterion = torch.nn.L1Loss()  # torch.nn.BCEWithLogitsLoss() #torch.nn.CrossEntropyLoss() # torch.nn.CrossEntropyLoss()
    criterion = nn.SmoothL1Loss(reduction='mean')
    scaler = torch.cuda.amp.GradScaler()
    eval_step = int(len(train_loader) / args.eval_times_per_epoch)
    start = end = time.time()

    # tbar = tqdm(train_loader, file=sys.stdout)
    losses = AverageMeter()
    preds = []
    labels = []

    # adversial
    attacker = None
    if cfg.attacker is not None:
        attacker = cfg.attacker(model)

    def eval_fn(_best_val_score, s):
        y_val, y_pred = validate(model, val_loader)
        score = -mcrmse(y_val, y_pred, len(target_columns))
        if score > _best_val_score:
            s = "[Best] " + s
            _best_val_score = score
            if not (args.is_experiment_stage and args.on_kaggle):
                torch.save(model.state_dict(),
                           os.path.join(output_path, f"{model_abbr}_best_F{fold}.bin"))
        logger.info("{}, Val score: {:.4f}".format(s, score))
        return round(_best_val_score, 4)

    for step, data in enumerate(train_loader):
        inputs, target = read_data(data)

        with torch.cuda.amp.autocast(enabled=cfg.apex):
            # print(inputs[0].size(), inputs[1].size())
            pred = model(*inputs)
            # print(pred.size(), target.size())
            loss = criterion(pred, target)
            # ce
            # output: [batch_size, nb_classes, *]
            # target [batch_size, *]
        if cfg.accumulation_steps > 1:
            loss = loss / cfg.accumulation_steps
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        if isinstance(attacker, FGM):
            attacker.attack()
            with torch.cuda.amp.autocast(enabled=CFG.apex):
                pred = model(*inputs)
                loss_adv = criterion(pred, target)
                loss_adv.backward()
            attacker.restore()

        if (step + 1) % cfg.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        losses.update(loss.detach().cpu().item(), cfg.batch_size)
        preds.append(pred.detach().cpu().numpy().ravel())
        labels.append(target.detach().cpu().numpy().ravel())

        # from https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train/notebook
        if (step + 1) % cfg.print_freq == 0 or step == (len(train_loader) - 1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch, step, len(train_loader),
                          remain=timeSince(start, float(step + 1) / len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))

        # todo accumulation
        # https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train/notebook

        if (step + 1) % eval_step == 0:
            info_str = f"epoch{epoch}, fold{fold}, step {step + 1} , Loss {'%.4f' % losses.avg}"
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


def print_info(_cfg=None):
    if _cfg is None:
        _cfg = cfg
    logger.info("*" * 15 + "  Info  " + "*" * 15)
    for k, v in cfg.__dict__.items():
        logger.info(f"\t{k}: {v}")


@timeit(logger)
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
    # save fold
    if not os.path.exists(oof_train_path):
        train.to_pickle(oof_train_path)

    # logger.info(train[train["kfold"]==1].head(10))

    ##
    best_scores = []
    for f in range(args.n_folds):
        best_val_score = train_fold(train, fold=f)
        best_scores.append(round(best_val_score, 4))
        if args.is_experiment_stage and (args.n_exp_stop_fold is not None and args.n_exp_stop_fold - 1 == f):
            break
    cv_score = round(float(np.mean(best_scores)), 4)
    logger.info("**** Best score in every fold: " + str(best_scores))
    logger.info("**** Best score Mean " + str(cv_score))

    if cfg.oof:
        oof_pipeline()

    def save_state():
        state_series = get_state_series()
        series = state_series
        series["scores"] = str(best_scores)
        series["CV_score"] = cv_score
        series["time"] = datetime.now().strftime("%m-%d %H:%M")
        output_df = pd.DataFrame([series])
        save_path = os.path.join(theme_path, "states.csv")
        if os.path.exists(save_path):
            df = pd.read_csv(save_path)
            output_df = pd.concat([df, output_df], axis=0)
        logger.info("save state.")
        output_df.to_csv(save_path, index=False)

    save_state()
    return cv_score


def get_logits_fold(test, test_model_path, is_oof=False):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    test_ds = cfg.MyDataset(test, model_name_or_path=args.model_name_or_path,
                            total_max_len=cfg.total_max_len,
                            columns=target_columns)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=args.n_workers,
                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, padding='longest'),
                             pin_memory=False, drop_last=False)

    logits = []
    for fold in range(args.n_folds):
        pth = f"{test_model_path}/{model_abbr}_best_F{fold}.bin"
        # load model
        s = time.time()
        model = cfg.MyModel(args.model_name_or_path, cfg, logger=logger).cuda()
        if cfg.parallel:
            model = nn.DataParallel(model)
        model.load_state_dict(torch.load(pth))
        logger.info(f"Load model fold {fold} cost time: {round(time.time() - s, 2)}s")

        _, pred = validate(model, test_loader)

        if is_oof:
            mask = np.array(test["kfold"] == fold)
            pred = (mask[:, None] * pred.reshape(-1, len(target_columns))).ravel()
            print("shape:", mask.shape, sum(mask))
        logits.append(pred)
        del model
        gc.collect()
    division = 1 if is_oof else args.n_folds
    logit_preds = sum(np.array(logits)) / division  # sum(a) equals sum(a, axis=0)
    return logit_preds

def get_path():
    cfg_path = args.cfg_path
    test_model_path = args.test_model_path
    if test_model_path is None:
        test_model_path = output_path
    if cfg_path is None:
        cfg_path = os.path.join(test_model_path, "cfg.bin")
    return cfg_path, test_model_path


def oof_pipeline():
    cfg_path, test_model_path = get_path()
    load_cfg(cfg_path)
    logger.info("* OOF model path: " + test_model_path)
    print_info()
    # read data
    train = pd.read_pickle(oof_train_path)
    # pred
    logit_preds = get_logits_fold(train, test_model_path, is_oof=True)
    class_preds = np.array([x for x in logit_preds]).reshape(-1, len(target_columns))
    train[[f"pred_{c}" for c in target_columns]] = class_preds
    train.to_pickle(oof_output_path)
    oof()


def test_pipeline():
    cfg_path, test_model_path = get_path()
    load_cfg(cfg_path)
    logger.info("* Test model path: " + test_model_path)
    print_info()
    test = pd.read_csv(args.test_path)
    test = fit_data(test)
    test[[target_columns]] = 0

    logit_preds = get_logits_fold(test, test_model_path)
    class_preds = np.array([x for x in logit_preds]).reshape(-1, len(target_columns))

    submission = pd.read_csv(os.path.join(os.path.dirname(args.test_path), "sample_submission.csv"))
    submission[target_columns] = class_preds
    submission_pth = os.path.join(os.path.dirname(args.test_path), "submission.csv")
    if args.on_kaggle:
        submission_pth = "/kaggle/working/submission.csv"
    submission.to_csv(submission_pth, index=False)


def exp_pipeline():
    global output_path, logger

    def set_params(params: dict):
        for p, v in params.items():
            cfg.__setattr__(p, v)

    def set_params_and_train(params):
        set_params(params)
        val = train_pipeline()
        clear()
        return val

    def clear():
        global output_path, logger
        remove_handler(logger)
        output_path, logger = set_args()

    def greedy_optimize(meta):
        d = {}
        best_score = -1e9
        for pivot_col in meta.keys():
            if len(meta[pivot_col]) == 1:
                # cfg.__setattr__(pivot_col, meta[pivot_col][0])
                d[pivot_col] = meta[pivot_col][0]
                clear()
            else:
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
                # 'n_unit': trial.suggest_int("n_unit", 4, 18)
            }
            val = set_params_and_train(meta)
            return val

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=30)

    # optuna_optimize()
    meta = {
        "lr": [5e-6, 1e-5],  # 尝试过 [1e-5, 2e-5, 3e-5, 4e-5]  1e-5明显改进
        "attacker": [FGM, None],
        # "is_bert_dp": [False, True],
        # "pooler": [MeanPooling],  # [AttentionPooling, MeanPooling],
        # "fc_dropout_rate": [0, 0.1, 0.15, 0.25, 0.3],
        "reinit_layer_num": [0, 1, 2]
        # "pooling_layers": [3, 2, 1],
        # [True, False], # False明显改进(再尝试一下)
    }
    greedy_optimize(meta)


def main():
    if args.is_train:
        logger.info("*" * 8 + "TRAIN STAGE" + "*" * 8)
        train_pipeline()
    if args.is_oof:
        logger.info("*" * 8 + "TEST STAGE" + "*" * 8)
        oof_pipeline()
    if args.is_test:
        logger.info("*" * 8 + "TEST STAGE" + "*" * 8)
        test_pipeline()
    if args.is_experiment_stage:
        exp_pipeline()


if __name__ == "__main__":
    main()
    # save_cfg("outputs/kaggle-ELL/exp65")

