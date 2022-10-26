import torch.nn.functional as F
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, \
    AutoModelForSequenceClassification, AutoConfig
from .pooling.pooling import *


# set position embeddings, dropout prob by config
class Rank1MarkdownModel(nn.Module):
    def __init__(self, name, num_classes=1, pretrained=True):
        super(MarkdownModel, self).__init__()
        self.config = AutoConfig.from_pretrained(name)
        self.config.attention_probs_dropout_prob = 0.
        self.config.hidden_dropout_prob = 0.
        self.config.max_position_embeddings = 4096 * 2
        # self.config.output_hidden_states = True
        if pretrained:
            self.encoder = AutoModel.from_pretrained(name, config=self.config, ignore_mismatched_sizes=True)
        else:
            self.encoder = AutoModel.from_config(self.config)
        self.in_dim = self.encoder.config.hidden_size
        self.bilstm = nn.LSTM(self.in_dim, self.in_dim, num_layers=1,
                              dropout=self.config.hidden_dropout_prob, batch_first=True,
                              bidirectional=True)
        self.last_fc = nn.Linear(self.in_dim * 2, num_classes)
        torch.nn.init.normal_(self.last_fc.weight, std=0.02)

    def forward(self, x, mask):
        x = self.encoder(x, attention_mask=mask)["last_hidden_state"]
        x, _ = self.bilstm(x)
        out = self.last_fc(x)
        out = out.squeeze(-1)
        return out


### https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently
class WeightedLayerPooling(torch.nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else torch.nn.Parameter(
            torch.tensor([1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float)
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average


class MarkdownModel(nn.Module):
    def __init__(self, model_path, logger=None, layer=5, verbose=False):
        super(MarkdownModel, self).__init__()
        self.use_classification_layer = False
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.update({'output_hidden_states': True})
        if self.use_classification_layer:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        else:
            self.model = AutoModel.from_pretrained(model_path, config=self.config)

        # print("hidden_sizes:", vars(self.config), type(self.config))
        hidden_size = self.config.hidden_size
        # print("hidden_size:", hidden_size)

        self.top = nn.Linear(hidden_size, 2)
        self.lm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        self.ms_dropout = [torch.nn.Dropout(x / 10) for x in range(layer)]
        self.dp = torch.nn.Dropout(0.2)
        self.pooler = WeightedLayerPooling(self.config.num_hidden_layers, layer_start=9, layer_weights=None)
        self.fc = torch.nn.Linear(self.config.hidden_size, 1)
        if logger is not None and verbose:
            logger.info("Model embedding size:" + str(self.model.embeddings.word_embeddings.weight.data.shape))
        # nn.init.xavier_uniform_(self.top.tensor, gain=nn.init.calculate_gain('relu'))

    def forward(self, ids, mask):
        if self.use_classification_layer:
            x = self.model(ids, mask).logits
            return x
        out_e = self.model(ids, mask)
        out = torch.stack(out_e["hidden_states"])
        out = self.pooler(out)
        for i, fc_dp in enumerate(self.ms_dropout):
            if i == 0:
                outputs = self.fc(fc_dp(out[:, 0]))
            else:
                outputs += self.fc(fc_dp(out[:, 0]))
        outputs = self.fc(self.dp(out[:, 0]))
        #
        return outputs


# weighted pool
class ELLModel(nn.Module):
    def __init__(self, model_path, logger=None, layer=5, verbose=False):
        super(ELLModel, self).__init__()
        self.use_classification_layer = False
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.update({'output_hidden_states': True})
        if self.use_classification_layer:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        else:
            self.model = AutoModel.from_pretrained(model_path, config=self.config)

        # print("hidden_sizes:", vars(self.config), type(self.config))
        hidden_size = self.config.hidden_size
        # print("hidden_size:", hidden_size)

        self.top = nn.Linear(hidden_size, 2)
        self.lm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        self.ms_dropout = [torch.nn.Dropout(x / 10) for x in range(layer)]
        self.dp = torch.nn.Dropout(0.2)
        self.pooler = WeightedLayerPooling(self.config.num_hidden_layers, layer_start=9, layer_weights=None)
        self.fc = torch.nn.Linear(self.config.hidden_size, 6)
        if logger is not None and verbose:
            logger.info("Model embedding size:" + str(self.model.embeddings.word_embeddings.weight.data.shape))
        # nn.init.xavier_uniform_(self.top.tensor, gain=nn.init.calculate_gain('relu'))

    def forward(self, ids, mask):
        if self.use_classification_layer:
            x = self.model(ids, mask).logits
            return x
        out_e = self.model(ids, mask)
        out = torch.stack(out_e["hidden_states"])
        out = self.pooler(out)
        for i, fc_dp in enumerate(self.ms_dropout):
            if i == 0:
                outputs = self.fc(fc_dp(out[:, 0]))
            else:
                outputs += self.fc(fc_dp(out[:, 0]))
        outputs = self.fc(self.dp(out[:, 0]))
        print("outputs size:", outputs.size())
        return outputs


# mean-max-pooling
class ELLModelv2(nn.Module):
    def __init__(self, model_path, logger=None, layer=5, verbose=False):
        super(ELLModelv2, self).__init__()
        self.use_classification_layer = False
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.update({'output_hidden_states': True})
        if self.use_classification_layer:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        else:
            self.model = AutoModel.from_pretrained(model_path, config=self.config)

        # print("hidden_sizes:", vars(self.config), type(self.config))
        hidden_size = self.config.hidden_size
        # print("hidden_size:", hidden_size)

        self.top = nn.Linear(hidden_size, 2)
        self.lm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        self.dp = torch.nn.Dropout(0.2)
        self.pooler = MeanPooling()
        self.fc = torch.nn.Linear(hidden_size, 6)
        self._init_weights(self.fc)
        if logger is not None and verbose:
            logger.info("Model embedding size:" + str(self.model.embeddings.word_embeddings.weight.data.shape))
        # nn.init.xavier_uniform_(self.top.tensor, gain=nn.init.calculate_gain('relu'))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, ids, mask):
        if self.use_classification_layer:
            x = self.model(ids, mask).logits
            return x
        out_e = self.model(ids, mask)["hidden_states"][-1]  # (b, l, h)  -1 represents the last hidden layer
        # out = torch.stack(out_e["hidden_states"])
        # print("out size:", out_e.size())
        out = self.pooler(out_e, mask)
        # print("out size:", out.size())
        outputs = self.fc(out)
        # print("outputs size:", outputs.size())
        return outputs

