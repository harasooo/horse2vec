import torch.nn as nn


class CustomMSELoss(nn.Module):
    def __init__(self, ignored_index: int, size_average=None, reduce=None) -> None:
        super(CustomMSELoss, self).__init__()

    def _mse_loss(self, input, target, mask):
        out = (input[~mask] - target[~mask]) ** 2
        return out.mean()

    def forward(self, input, target, mask):
        return self._mse_loss(input.view(-1, 18), target.view(-1, 18), mask)
