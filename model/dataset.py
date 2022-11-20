from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer, BertTokenizer
import numpy as np
import collections
import random
from .pretrain.mlm import mask_tokens_ngram
from .padding.padding import feat_truncation


class ELLDataset(Dataset):
    def __init__(self, df, model_name_or_path, total_max_len, md_max_len, columns=["label"]):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.columns = columns

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path)  # BertTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = row.text

        inputs = self.tokenizer.encode_plus(
            row.text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.total_max_len,
            padding="max_length",
            #return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        ids = torch.LongTensor(ids)   # size (L)
        mask = torch.LongTensor(mask)  # size (L)
        # print("id size:", ids.size(), "mask size:", mask.size())

        target = torch.FloatTensor([row[column] for column in self.columns])

        assert len(ids) == self.total_max_len
        assert len(mask) == self.total_max_len

        return ids, mask, target

    def __len__(self):
        return self.df.shape[0]


class ELLDatasetNoPadding(Dataset):
    def __init__(self, df, model_name_or_path, total_max_len, columns=["label"], is_pretrain=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.columns = columns
        self.is_pretrain = is_pretrain
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path)  # BertTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.text,
            return_tensors=None,
            add_special_tokens=True,
            # return_token_type_ids=True,
            max_length=self.total_max_len,
            padding="max_length" if self.is_pretrain else False,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']


        # print("id size:", ids.size(), "mask size:", mask.size())

        if self.is_pretrain:
            # no need to truncate, because input labels are out of position range itself,
            sep_token = 102
            for column in self.columns:
                ids += [sep_token, round(row[column]*2)+1]
                mask += [1, 1]
            ids, target = mask_tokens_ngram(ids, self.tokenizer.vocab_size)
            target = torch.LongTensor(target)
        else:
            target = torch.FloatTensor([row[column] for column in self.columns])
        ids = torch.LongTensor(ids)  # size (L)
        mask = torch.LongTensor(mask)  # size (L)

        return {"input_ids": ids, "attention_mask": mask, "labels": target}

    def __len__(self):
        return self.df.shape[0]


class ELLDatasetRandomTruncation(Dataset):
    def __init__(self, df, model_name_or_path, total_max_len, columns=["label"], is_pretrain=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.columns = columns
        self.is_pretrain = is_pretrain
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path)  # BertTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.text,
            return_tensors=None,
            add_special_tokens=True,
            # return_token_type_ids=True,
            # max_length=self.total_max_len,
            padding=False,
            # truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']


        # print("id size:", ids.size(), "mask size:", mask.size())

        if self.is_pretrain:
            # no need to truncate, because input labels are out of position range itself,
            sep_token = 102
            for column in self.columns:
                ids += [sep_token, round(row[column]*2)+1]
                mask += [1, 1]
            ids, target = mask_tokens_ngram(ids, self.tokenizer.vocab_size)
            target = torch.LongTensor(target)
        else:
            target = torch.FloatTensor([row[column] for column in self.columns])
        ids = torch.LongTensor(ids)  # size (L)
        # print("t:",ids.size())
        mask = torch.LongTensor(mask)  # size (L)
        ids, mask = feat_truncation(ids, mask, self.total_max_len)
        # print("a:",ids.size())
        return {"input_ids": ids, "attention_mask": mask, "labels": target}

    def __len__(self):
        return self.df.shape[0]


