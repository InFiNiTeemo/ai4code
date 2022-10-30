import json
from pathlib import Path
from dataset import *
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from model import *
from tqdm import tqdm
import sys, os
from metrics import *
import torch
import argparse
from sklearn.metrics import f1_score, classification_report, accuracy_score
#from focal_loss.focal_loss import FocalLoss
from utils.focal_loss import FocalLoss
from utils.preprocess import get_group_dict, get_code_dict
import random
import gc
import pickle
import logging


parser = argparse.ArgumentParser(description='Process some arguments')
parser.add_argument('--seed', type=int, default=37)
parser.add_argument('--fold', type=str, default="N")
parser.add_argument('--model_name_or_path', type=str, default='microsoft/codebert-base-vocabplus') # #'WENGSYX/Deberta-Chinese-Large')# 'hfl/chinese-macbert-base') #'microsoft/codebert-base')
#parser.add_argument('--train_mark_path', type=str, default='./data/train_mark.csv')
#parser.add_argument('--train_features_path', type=str, default='./data/train_fts.json')
#parser.add_argument('--val_mark_path', type=str, default='./data/val_mark.csv')
#parser.add_argument('--val_features_path', type=str, default='./data/val_fts.json')
parser.add_argument('--train_path', type=str, default="./data/public/train/block1.csv")
parser.add_argument('--pair_path', type=str, default="./data/public/train/block1.bin")
parser.add_argument('--val_path', type=str, default="./data/yb_train.csv")
parser.add_argument('--test_path', type=str, default="./data/test1.csv")
parser.add_argument('--md_max_len', type=int, default=64)
parser.add_argument('--total_max_len', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--accumulation_steps', type=int, default=4)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--is_train', action='store_true')
parser.add_argument('--no_train', dest='is_train', action='store_false')
parser.set_defaults(is_train=True)



args = parser.parse_args()
if not os.path.exists("./outputs"):
    os.mkdir("./outputs")
data_dir = Path('..//input/')

#train_df_mark = pd.read_csv(args.train_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
#train_fts = json.load(open(args.train_features_path))
#val_df_mark = pd.read_csv(args.val_mark_path).drop("parent_id", axis=1).dropna().reset_index(drop=True)
#val_fts = json.load(open(args.val_features_path))

def get_model_abbr(model_name):
    # todo prefix match
    model_abbr_dict = {
        'microsoft/codebert-base-vocabplus': "codebertP",
        'microsoft/codebert-base': "codebert",
        "hfl/chinese-macbert-large":"macL",
        'hfl/chinese-macbert-base':"mac",
        'microsoft/deberta-base':"deb",
        "fnlp/bart-large-chinese":"bartL",
        "hfl/chinese-roberta-wwm-ext":"rob",
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




def split_dataset(df, frac=0.9):
    logger.info("**split dataset**")
    if isinstance(df, dict):
        return dict(df.items()[:int(len(df) * frac)]), dict(df.items()[int(len(df) * frac):])
    elif isinstance(df, list):
        return df[:int(len(df) * frac)], df[int(len(df) * frac):]
    index = df.sample(frac=frac).index
    train_df = df[df.index.isin(index)]
    val_df = df[~df.index.isin(index)]
    return train_df, val_df


def get_hidden_size(s:str):
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
seed_everything(args.seed)

# order_df = pd.read_csv("../input/train_orders.csv").set_index("id")
#df_orders = pd.read_csv(
#    data_dir / 'train_orders.csv',
#    index_col='id',
#    squeeze=True,
#).str.split()
is_train = args.is_train #.False #True #False
train_loader, val_loader = None, None

test_df = pd.read_csv(args.test_path, sep='\t', error_bad_lines=False)
test_df['label'] = 0
test_ds = LanguageMistakeDataset(test_df, model_name_or_path=args.model_name_or_path, md_max_len=args.md_max_len,
                         total_max_len=args.total_max_len)
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                        pin_memory=False, drop_last=False)

logger = get_logger(filename=f"./log/train_{get_model_abbr(args.model_name_or_path)}")


def read_data(data):
    return tuple(d.cuda() for d in data[:-1]), data[-1].cuda()


def validate(model, val_loader):
    model.eval()

    tbar = tqdm(val_loader, file=sys.stdout)

    preds = []
    labels = []

    with torch.no_grad():
        for idx, data in enumerate(tbar):
            inputs, target = read_data(data)

            with torch.cuda.amp.autocast():
                pred = model(*inputs).argmax(-1)

            preds.append(pred.detach().cpu().numpy().ravel())
            labels.append(target.detach().cpu().numpy().ravel())

    return np.concatenate(labels), np.concatenate(preds)


def test(model, test_loader, test_df):
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
    test_df['label'] = y_pred
    test_df = test_df[['id', 'label']]
    test_df.to_csv("submission.csv", index=False, sep='\t')
    return y_pred


def infer_ensemble(model_list_name, test_loader, test_df):
    i = 0
    l = len(test_df)
    pred = np.zeros(l)
    for model_path in model_list_name:
        model = None
        if i < 5:
            model = MarkdownModel('hfl/chinese-macbert-large', 1024)
        else:
            model = MarkdownModel('hfl/chinese-macbert-base', 768)
        model.load_state_dict(torch.load(f"./outputs/{model_path}"))
        model.cuda()
        cur_pred = test(model, test_loader, test_df)
        if i < 5:
            pred += cur_pred / 5
        else:
            pred += cur_pred * 0.7 / 10
        i += 1
        del model
        gc.collect()
    print(pred)
    f = lambda x: 1 if x >= 0.5 else 0
    fv = np.vectorize(f)
    pred = fv(pred)
    test_df['label'] = pred
    test_df = test_df[['id', 'label']]
    test_df.to_csv("submission_ensemble.csv", index=False, sep='\t')


def train(model, optimizer, scheduler, train_loader, val_loader, epoch, val_pairs, block):
    # np.random.seed(0)
    # Creating optimizer and lr schedulers


    # criterion = FocalLoss(class_num=2, alpha=torch.FloatTensor([0.7, 0.3]))  # num in test 0 : 1 = 0.385: 0.615
    criterion = torch.nn.CrossEntropyLoss() #torch.nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler()

    tbar = tqdm(train_loader, file=sys.stdout)
    loss_list = []
    preds = []
    labels = []

    for idx, data in enumerate(tbar):
        inputs, target = read_data(data)

        with torch.cuda.amp.autocast():
            pred = model(*inputs)
            # print(pred.size(), target.size())
            loss = criterion(pred, target[:, 0])
            # ce
            # output: [batch_size, nb_classes, *]
            # target [batch_size, *]
        scaler.scale(loss).backward()
        if idx % args.accumulation_steps == 0 or idx == len(tbar) - 1:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        #if idx % 10000 == 0:
        #    torch.save(model.state_dict(), f"./outputs/yb_model_{idx}.bin")

        loss_list.append(loss.detach().cpu().item())
        preds.append(pred.detach().cpu().numpy().ravel())
        labels.append(target.detach().cpu().numpy().ravel())

        avg_loss = np.round(np.mean(loss_list), 4)

        tbar.set_description(f"Epoch {epoch + 1} Loss: {avg_loss} lr: {scheduler.get_last_lr()}")

        #if idx > 50:
        #    break

    y_val, y_pred = validate(model, val_loader)

    # y_dummy = val_df.sort_values("pred").groupby('id')['cell_id'].apply(list)
    val_label = [label for p1, p2, label in val_pairs]
    score = f1_score(val_label, y_pred, average='macro')
    logger.info(f"epoch{epoch}, block{block}")
    logger.info("Preds score:"+ str(score))
    logger.info(classification_report(val_label, y_pred))
    print(pd.Series(val_label).value_counts())
    print("\n", pd.Series(y_pred).value_counts())

    torch.save(model.state_dict(), f"./outputs/{get_model_abbr(args.model_name_or_path)}_F{block}_E{epoch+1}_{score}.bin")

    return model


def main():
    model = MarkdownModel(args.model_name_or_path, get_hidden_size(args.model_name_or_path), logger)
    model = model.cuda()
    model.train()


    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(args.epochs * 13749 * 100 / args.accumulation_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5,
                      correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    num_warmup_steps = min(num_train_optimization_steps * 0.02, 10000)
    is_load = False
    if is_load:
        model.load_state_dict(torch.load("./outputs/codebert_F42_E1_0.9100168554389882.bin"))
        num_warmup_steps = 0

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
    # val
    val_pairs = pickle.load(open(f"./data/public/train/block99.bin", "rb"))
    val_pairs = val_pairs[:50000]
    val_df = pd.read_csv(f"./data/public/train/block99.csv")
    val_code_dict = get_code_dict(val_df)
    val_group_dict = get_group_dict(val_df)
    del val_df
    val_ds = CloneDetectionDataset(val_pairs, val_code_dict, val_group_dict, model_name_or_path=args.model_name_or_path,
                                   md_max_len=args.md_max_len,
                                   total_max_len=args.total_max_len,
                                   logger=logger)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers,
                            pin_memory=False, drop_last=False)

    for e in range(args.epochs):
        for i in range(99):
            #if e == 0 and i <= 42:
            #    continue
            train_pairs = pickle.load(open(f"./data/public/train/block{i}.bin", "rb"))
            train_df = pd.read_csv(f"./data/public/train/block{i}.csv")
            code_dict = get_code_dict(train_df)
            group_dict = get_group_dict(train_df)
            # val_df = pd.read_csv(args.val_path, sep='\t', error_bad_lines=False)
            # train_pairs, val_pairs = split_dataset(pair_dict)  # pd.read_csv(args.val_path, sep='\t', error_bad_lines=False)
            train_ds = CloneDetectionDataset(train_pairs, code_dict, group_dict, model_name_or_path=args.model_name_or_path,
                                             md_max_len=args.md_max_len,
                                             total_max_len=args.total_max_len)

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                                      pin_memory=False, drop_last=True)


            # model.load_state_dict(torch.load("./outputs/macbert_pfocal_9_0.546272533000685.bin"))
            # model.load_state_dict(torch.load("./outputs/macbert_focal_5_0.825334437898871.bin"))
            model = train(model, optimizer, scheduler, train_loader, val_loader, epoch=e, val_pairs=val_pairs, block=i)




if is_train:
    main()
    #model.load_state_dict(torch.load("./outputs/macbert_pfocal_9_0.546272533000685.bin"))
    # model.load_state_dict(torch.load("./outputs/macbert_focal_5_0.825334437898871.bin"))
    # model = train(model, train_loader, val_loader, epochs=args.epochs)
else:
    model_list_name = ["robL_F0_E10_0.8087224411189796.bin", #"macL_E10_F4_0.8140642591255138.bin", "macL_E10_F3_0.8034712547147682.bin", "macL_E10_F2_0.806989913364951.bin", "macL_E10_F1_0.8041033810974407.bin", "macL_E10_F0_0.8119268320159683.bin",
                       ]#"mac_E10_F0_0.7920503285857519.bin", "mac_E10_F1_0.7873531553301534.bin", "mac_E10_F2_0.7930166735040224.bin", "mac_E10_F3_0.7894329689373656.bin", "mac_E10_F4_0.7972659183112555.bin"]

    #infer_ensemble(model_list_name, test_loader, test_df)
    model = MarkdownModel(args.model_name_or_path, get_hidden_size(args.model_name_or_path))
    model = model.cuda()
    model.load_state_dict(torch.load("./outputs/robL_F0_E10_0.8087224411189796.bin"))
    test(model, test_loader, test_df)


# 0.7620  macbert_pfocal_10_0.8121012861574644.bin    focal_coefficient 0.7 0.3
# 0.7420  macbert_pfocal_10_0.822280030947326.bin   focal_coefficient 0.385 0.615
# 0.759   ensemble 10 model   focal_coefficient 0.615 0.385
# 0.760   ensemble 5 mac large model   focal_coefficient 0.615 0.385
