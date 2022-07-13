from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer

from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer


class MarkdownDataset(Dataset):

    def __init__(self, order_df, df, model_name_or_path, total_max_len, md_max_len, fts):
        super().__init__()
        self.order_df = order_df
        self.df = df.reset_index(drop=True)
        self.md_max_len = md_max_len
        self.total_max_len = total_max_len  # maxlen allowed by model config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.fts = fts

    def __getitem__(self, index):
        row = self.order_df.iloc[index]
        cells = row.cell_shuffle

        inputs = self.tokenizer.encode_plus(
            [row.source[cell_id] for cell_id in cells],
            None,
            add_special_tokens=True,
            max_length=self.self.total_max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True
        )

        ids = torch.LongTensor(inputs['ids'])

        mask = torch.LongTensor(inputs['attention_mask'])

        md_mask = torch.LongTensor(row.md_mask)

        assert len(ids) == self.total_max_len

        return ids, mask, row.permutation, md_mask

    def __len__(self):
        return self.df.shape[0]