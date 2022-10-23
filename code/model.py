import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModelForSequenceClassification, AutoConfig



HIDDEN_SIZE = 768


class MarkdownModel(nn.Module):
    def __init__(self, model_path, hidden_size=None, logger=None):
        super(MarkdownModel, self).__init__()
        self.use_classification_layer = False
        if self.use_classification_layer:
            self.config = AutoConfig.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        else:
            self.model = AutoModel.from_pretrained(model_path)
        if hidden_size is None:
            hidden_size = HIDDEN_SIZE
        self.top = nn.Linear(hidden_size, 2)
        self.lm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        if logger is not None:
            logger.info("Model embedding size:" + str(self.model.embeddings.word_embeddings.weight.data.shape))
        # nn.init.xavier_uniform_(self.top.tensor, gain=nn.init.calculate_gain('relu'))

    def forward(self, ids, mask):
        if self.use_classification_layer:
            x = self.model(ids, mask).logits
            return x
        #print("x size:", x.size())
        # print("ids:", ids, mask)
        x = self.model(ids, mask)[0][:, 0, :]
        x = self.top(x)
        # x = torch.cat((x[:, 0, :], fts), 1)
        # x = self.lm(x)
        #
        return x





