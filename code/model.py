import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup


HIDDEN_SIZE = 768


class MarkdownModel(nn.Module):
    def __init__(self, model_path):
        super(MarkdownModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(769, 1)

    def forward(self, ids, mask, fts):
        x = self.model(ids, mask)[0]
        x = torch.cat((x[:, 0, :], fts), 1)
        x = self.top(x)
        return x


class BersonModel(nn.Module):
    def __init__(self, model_path):
        super(BersonModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.top = nn.Linear(769, 1)

        self.dropout = nn.Dropout(0.1)
        self.decoder = nn.LSTM(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE*2, num_layers=2, batch_first=True)

    def forward(self, ids, mask, order, type):
        """
        :param ids:
        :param mask:
        :param orders:
        :param type: 0 for code, 1 for md
        :return:
        """
        x = self.model(ids, mask)[0]

        ## concat
        x

        # lstm
        hcn = None
        for t in range():
            output, hcn = self.decoder(x, hcn) # hcn = (hidden_state, cell_state)
        # loss
        loss = ...

        ## sentence order

        ## coherence
        return loss