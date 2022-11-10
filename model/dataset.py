from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer, BertTokenizer
import numpy as np
import collections
import random
from .pretrain.mlm import mask_tokens_ngram


class MarkdownDataset(Dataset):

    def __init__(self, order_df, df, model_name_or_path, total_max_len, md_max_len, fts, logger=None):
        super().__init__()

        self.order_df = order_df
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if logger is not None:
            logger.info("tokenizer vocab size:" + str(len(self.tokenizer)))

        self.fts = fts

    def __getitem__(self, index):
        row = self.df.iloc[index]

        inputs = self.tokenizer.encode_plus(
            row.source,
            None,
            add_special_tokens=True,
            max_length=self.md_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )
        code_inputs = self.tokenizer.batch_encode_plus(
            [str(x) for x in self.fts[row.id]["codes"]],
            add_special_tokens=True,
            max_length=23,
            padding="max_length",
            truncation=True
        )
        n_md = self.fts[row.id]["total_md"]
        n_code = self.fts[row.id]["total_md"]
        if n_md + n_code == 0:
            fts = torch.FloatTensor([0])
        else:
            fts = torch.FloatTensor([n_md / (n_md + n_code)])

        ids = inputs['input_ids']
        for x in code_inputs['input_ids']:
            ids.extend(x[:-1])
        ids = ids[:self.total_max_len]
        if len(ids) != self.total_max_len:
            ids = ids + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(ids))
        ids = torch.LongTensor(ids)

        mask = inputs['attention_mask']
        for x in code_inputs['attention_mask']:
            mask.extend(x[:-1])
        mask = mask[:self.total_max_len]
        if len(mask) != self.total_max_len:
            mask = mask + [self.tokenizer.pad_token_id, ] * (self.total_max_len - len(mask))
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len

        return ids, mask, fts, torch.FloatTensor([row.pct_rank])

    def __len__(self):
        return self.df.shape[0]


class CloneDetectionDataset(Dataset):
    def __init__(self, pairs, code_dict, group_dict, model_name_or_path, total_max_len, md_max_len, logger=None):
        super().__init__()
        # self.df = df.reset_index(drop=True)
        self.pairs = pairs
        self.code_dict = code_dict
        self.group_dict = group_dict
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path)  # BertTokenizer.from_pretrained(model_name_or_path)

        if logger is not None:
            logger.info("tokenizer vocab size:" + str(len(self.tokenizer)))

    def __getitem__(self, index):
        with_part_of_speech = False

        p1, p2, label = self.pairs[index]
        # print("pairs:", p1, p2)
        t = [self.code_dict[p1], self.code_dict[p2]]
        # label = (self.group_dict[p1] == self.group_dict[p2])

        inputs = [
            self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True
            ) for text in t
        ]

        test = False
        if test:
            print("raw text:", )
            # add_special_tokens=True is set by default
            text_enc = self.tokenizer.encode(t[0])

            for tok in text_enc:
                print(tok, self.tokenizer.decode(tok))

        # print(inputs)
        ids = [self.tokenizer.cls_token_id] + inputs[0]['input_ids'] + inputs[1]['input_ids']
        ids = ids[:self.total_max_len]
        ids = ids + [self.tokenizer.pad_token_id] * (self.total_max_len - len(ids))

        # ids = inputs['input_ids']
        mask = [1] + inputs[0]['attention_mask'] + inputs[1]['attention_mask']
        mask = mask[:self.total_max_len]
        mask = mask + [0] * (self.total_max_len - len(mask))

        # print(len(ids), len(mask))

        ids = torch.LongTensor(ids)
        mask = torch.LongTensor(mask)
        target = torch.LongTensor([label])



        assert len(ids) == self.total_max_len
        assert len(mask) == self.total_max_len

        return ids, mask, target

    def __len__(self):
        return len(self.pairs)


class LanguageMistakeDataset(Dataset):
    def __init__(self, df, model_name_or_path, total_max_len, md_max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path)  # BertTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        with_part_of_speech = False

        row = self.df.iloc[index]

        text = row.text

        inputs = self.tokenizer.encode_plus(
            row.text,
            None,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )

        # print(inputs)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        target = torch.LongTensor([row.label])

        if with_part_of_speech:
            # text = row.text + self.tokenizer.sep_token_id + row.part_of_speech

            p_inputs = self.tokenizer.encode_plus(
                row.part_of_speech,
                None,
                add_special_tokens=True,
                max_length=39,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True
            )

            ids = ids + [self.tokenizer.sep_token_id] + p_inputs['input_ids']
            mask = mask + [self.tokenizer.sep_token_id] + p_inputs['attention_mask']

        ids = torch.LongTensor(ids)
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len
        assert len(mask) == self.total_max_len

        return ids, mask, target

    def __len__(self):
        return self.df.shape[0]


class ClassificationDataset(Dataset):
    def __init__(self, df, model_name_or_path, total_max_len, md_max_len):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path)  # BertTokenizer.from_pretrained(model_name_or_path)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text = row.text

        inputs = self.tokenizer.encode_plus(
            row.text,
            None,
            add_special_tokens=True,
            max_length=self.total_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )

        # print(inputs)
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        target = torch.FloatTensor([row.label])

        with_part_of_speech = False
        if with_part_of_speech:
            # text = row.text + self.tokenizer.sep_token_id + row.part_of_speech

            p_inputs = self.tokenizer.encode_plus(
                row.part_of_speech,
                None,
                add_special_tokens=True,
                max_length=39,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True
            )

            ids = ids + [self.tokenizer.sep_token_id] + p_inputs['input_ids']
            mask = mask + [self.tokenizer.sep_token_id] + p_inputs['attention_mask']

        ids = torch.LongTensor(ids)
        mask = torch.LongTensor(mask)

        assert len(ids) == self.total_max_len
        assert len(mask) == self.total_max_len

        return ids, mask, target

    def __len__(self):
        return self.df.shape[0]


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



