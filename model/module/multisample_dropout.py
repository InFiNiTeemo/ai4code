import torch
import numpy as np
from torch import nn

# nn.Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:
class multilabel_dropout(nn.Module):
    # Multisample Dropout: https://arxiv.org/abs/1905.09788
    def __init__(self, HIGH_DROPOUT, HIDDEN_SIZE, n_classes):
        super(multilabel_dropout, self).__init__()
        self.high_dropout = torch.nn.Dropout(HIGH_DROPOUT)
        self.classifier = torch.nn.Linear(HIDDEN_SIZE, n_classes)

    def forward(self, out):
        return torch.mean(torch.stack([
            self.classifier(self.high_dropout(out))
            for _ in range(5)
        ], dim=0), dim=0)