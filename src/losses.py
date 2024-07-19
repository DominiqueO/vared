import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from typing import Optional


class RegCrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, gamma=1e-3, alpha=1, regularizer=None, model=None, weight: Optional[Tensor] = None,
                 size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.regularizer = regularizer
        self.alpha = alpha
        self.gamma = gamma
        self.model = model

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        ce_term = F.cross_entropy(input, target, weight=self.weight,
                                  ignore_index=self.ignore_index, reduction=self.reduction,
                                  label_smoothing=self.label_smoothing)
        reg_term = self.regularizer(self.model, self.alpha)
        return ce_term + self.gamma * reg_term

# Example regularizer function: L2 regularization
def regularizer(model, alpha=1):
    reg = torch.tensor(0.0, requires_grad=True)
    for param in model.parameters():
        if param.requires_grad:
            reg += (alpha * param**2) / (1 + alpha * param**2)
    return reg