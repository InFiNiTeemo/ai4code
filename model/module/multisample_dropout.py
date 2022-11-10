import torch
import numpy as np
from torch import nn

# nn.Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:
class multilabel_dropout():
    # Multisample Dropout: https://arxiv.org/abs/1905.09788
    def __init__(self, HIGH_DROPOUT, HIDDEN_SIZE, n_classes):
        self.high_dropout = torch.nn.Dropout(HIGH_DROPOUT)
        self.classifier = torch.nn.Linear(HIDDEN_SIZE * 2, n_classes)
    def forward(self, out):
        return torch.mean(torch.stack([
            self.classifier(self.high_dropout(p)(out))
            for p in np.linspace(0.1,0.5, 5)
        ], dim=0), dim=0)