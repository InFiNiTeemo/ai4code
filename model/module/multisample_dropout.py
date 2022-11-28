import torch
import numpy as np
from torch import nn

# nn.Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:
class multilabel_dropout(nn.Module):
    # Multisample Dropout: https://arxiv.org/abs/1905.09788
    def __init__(self, config, cfg, n_out_features):
        super(multilabel_dropout, self).__init__()
        self.high_dropout = torch.nn.Dropout(cfg.fc_dropout_rate)
        self.classifier = torch.nn.Linear(config.hidden_size, n_out_features)

    def forward(self, out):
        return torch.mean(torch.stack([
            self.classifier(self.high_dropout(out))
            for _ in range(5)
        ], dim=0), dim=0)

class multi_dropout(nn.Module):
    def __init__(self, cfg, n_out_features):
        super().__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(cfg.hidden_size, n_out_features)

    def forward(self, out):
        preds1 = self.classifier(self.dropout1(out))
        preds2 = self.classifier(self.dropout2(out))
        preds3 = self.classifier(self.dropout3(out))
        preds4 = self.classifier(self.dropout4(out))
        preds5 = self.classifier(self.dropout5(out))
        preds = (preds1 + preds2 + preds3 + preds4 + preds5) / 5
        return preds