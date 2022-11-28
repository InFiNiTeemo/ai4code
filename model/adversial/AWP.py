##############
# Code taken from this notebook :
# https://www.kaggle.com/code/skraiii/pppm-tokenclassificationmodel-train-8th-place
##############
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss


class AWP:
    def __init__(
            self,
            model,
            criterion: _Loss,
            optimizer,
            adv_param="weight",
            adv_lr=1e-3,
            adv_eps=0.2,
            start_epoch=0,
            adv_step=1,
            scaler=None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup = {}
        self.backup_eps = {}
        self.scaler = scaler

    def attack_backward(self, inputs, target, epoch):
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None

        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with torch.cuda.amp.autocast():
                pred = self.model(*inputs)
                adv_loss = self.criterion(pred, target)
            self.optimizer.zero_grad()
            self.scaler.scale(adv_loss).backward()

        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self, ):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}



class AWP_bad:
    def __init__(
        self,
        model: Module,
        criterion: _Loss,
        optimizer: Optimizer,
        apex: bool,
        adv_param: str="weight",
        adv_lr: float=1.0,
        adv_eps: float=0.01 # main para to adjust, default 0.2
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.apex = apex
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, label: Tensor) -> Tensor:
        with torch.cuda.amp.autocast(enabled=self.apex):
            self._save()
            self._attack_step()
            y_preds = self.model(*inputs)
            adv_loss = self.criterion(
                y_preds.view(-1, 1), label.view(-1, 1))
            mask = (label.view(-1, 1) != -1)
            adv_loss = torch.masked_select(adv_loss, mask).mean()
            self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}