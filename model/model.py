import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.utils import checkpoint
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, \
    AutoModelForSequenceClassification, AutoConfig
from .pooling.pooling import *
from .module.multisample_dropout import multilabel_dropout


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
        self.dp = torch.nn.Dropout(0.3)
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


# mean-max-pooling
class ELLModelv3(nn.Module):
    def __init__(self, model_path, logger=None, layer=5, verbose=False):
        super(ELLModelv3, self).__init__()
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

        self.lm = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        self.dp = torch.nn.Dropout(0.3)
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


class ELLModelv4(nn.Module):
    def __init__(self, model_path, cfg, logger=None, verbose=False):
        super(ELLModelv4, self).__init__()
        self.use_classification_layer = False
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.update({'output_hidden_states': True})
        self.config.max_position_embeddings = 512  # 对于一个competition不要改, 设为max_embedding的两倍
        hidden_size = self.config.hidden_size
        if not cfg.is_bert_dp:
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.

        # module
        if self.use_classification_layer:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        else:
            self.model = AutoModel.from_pretrained(model_path, config=self.config)
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.dp = torch.nn.Dropout(cfg.fc_dropout_rate)

        # pooler
        if cfg.pooler == AttentionPooling:
            self.pooler = cfg.pooler(hidden_size)
        else:
            self.pooler = cfg.pooler()
        self.pooling_layers = cfg.pooling_layers
        if isinstance(self.pooler, MeanMaxPooling):
            self.fc = torch.nn.Linear(2*hidden_size*self.pooling_layers, 6)
        else:
            self.fc = torch.nn.Linear(hidden_size*self.pooling_layers, 6)

        # init weights
        self.reinit_last_layers(cfg.reinit_layer_num)
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

    def reinit_last_layers(self, layer_num):
        if layer_num <= 0:
            return
        for module in self.model.encoder.layer[-layer_num:].modules():
            self._init_weights(module)

    def forward(self, ids, mask):
        if self.use_classification_layer:
            x = self.model(ids, mask).logits
            return x
        # out_e = self.model(ids, mask)["hidden_states"][-1]  # (b, l, h)  -1 represents the last hidden layer
        out_e = list(self.pooler(self.model(ids, mask)["hidden_states"][-i], mask) for i in range(1, self.pooling_layers+1))
        # print("out size:", out_e.size())
        out = torch.cat(out_e, 1)
        # print("out size:", out.size())
        outputs = self.fc(out)
        # print("outputs size:", outputs.size())
        return outputs

class ELLModelTest(nn.Module):
    def __init__(self, model_path, cfg, logger=None, verbose=False):
        super(ELLModelTest, self).__init__()
        self.use_classification_layer = False
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.update({'output_hidden_states': True})
        self.config.max_position_embeddings = 512  # 对于一个competition不要改, 设为max_embedding的两倍
        hidden_size = self.config.hidden_size
        if not cfg.is_bert_dp:
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.

        # module
        if self.use_classification_layer:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=self.config)
        else:
            self.model = AutoModel.from_pretrained(model_path, config=self.config)
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()


        # pooler
        if cfg.pooler == AttentionPooling:
            self.pooler = cfg.pooler(hidden_size)
        else:
            self.pooler = cfg.pooler()
        self.pooling_layers = cfg.pooling_layers
        if isinstance(self.pooler, MeanMaxPooling):
            linear_size = hidden_size*self.pooling_layers*2
        else:
            linear_size = hidden_size * self.pooling_layers

        out_features = len(cfg.target_columns)
        if cfg.fc == "multisample_dropout":
            self.fc = multilabel_dropout(cfg.fc_dropout_rate, linear_size, out_features)
        else:
            self.fc = nn.Linear(linear_size, out_features)

        # init weights
        self.reinit_last_layers(cfg.reinit_layer_num)
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

    def reinit_last_layers(self, layer_num):
        if layer_num <= 0:
            return
        for module in self.model.encoder.layer[-layer_num:].modules():
            self._init_weights(module)

    def forward(self, ids, mask):
        if self.use_classification_layer:
            x = self.model(ids, mask).logits
            return x
        # out_e = self.model(ids, mask)["hidden_states"][-1]  # (b, l, h)  -1 represents the last hidden layer


        # origin before 11.13
        # check一下 这里是否用错了
        # out_e = list(self.pooler(self.model(ids, mask)["hidden_states"][-i], mask) for i in range(1, self.pooling_layers+1))
        # now 11.13
        # 大家都这么用
        out_e = list(self.pooler(self.model(ids, mask)[i], mask) for i in range(0, self.pooling_layers))

        # print("out size:", out_e.size())
        out = torch.cat(out_e, 1)
        # print("out size:", out.size())
        outputs = self.fc(out)
        # print("outputs size:", outputs.size())
        return outputs

