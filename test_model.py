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
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import time
from utils.util import AverageMeter, split_dataset
from typing import Any
from functools import wraps

def get_logger(filename='test'):
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

logger = get_logger()


def test_model():
    model_name_or_path = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = ELLModelv2(model_name_or_path)
    model.eval()
    model = model.cuda()
    a = torch.ones(8, 30).long().cuda()
    mask = torch.ones(8, 30).long().cuda()
    b = model(a, mask)
    c = torch.ones(8)
    # print(help(tokenizer.pad))
    print(tokenizer.pad(c, max_length=20))
    # print(b)

def test_pandas():
    a = [1,2,3]
    c = pd.Series(a, index=['a', 'c', 'd'])
    d = pd.Series(a, index=['a', 'c', 'd'])
    df = pd.DataFrame([c,d])
    e = pd.DataFrame([pd.Series(a, index=['a', 'c', 'd'])])
    df = pd.concat([df, e], axis=0)
    print(df)


from dataclasses import dataclass, make_dataclass
@dataclass(init=True, repr=True, eq=True)
class CFG:
    MyDataset: Any = ELLDatasetNoPadding
    # model
    MyModel: Any = ELLModelv2  # ELLModelv2
    dropout_rate: float = 0.2
    pooler: Any = MeanPooling


cfg = CFG()

def show_config():
    print(cfg.__dict__)
    print(vars(cfg))
    print(cfg.MyModel)




def test_variable():
    for i in range(int(1e4)):
        print("")
    from datetime import datetime
    d = datetime.now().strftime("%m-%d %H:%M")
    print(d)
    return 3

if __name__ == "__main__":
    a = test_variable()
    print(a)