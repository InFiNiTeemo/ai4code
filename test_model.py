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
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import time
from utils.util import AverageMeter, split_dataset

def main():
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

main()