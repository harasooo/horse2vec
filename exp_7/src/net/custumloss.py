import torch.nn as nn


class CustomMSELoss(nn.Module):
    def __init__(self, ignored_index: int, size_average=None, reduce=None) -> None:
        super(CustomMSELoss, self).__init__()
        self.ignored_index = ignored_index

    def _mse_loss(self, input, target, ignored_index):
        mask = target == ignored_index
        out = (input[~mask] - target[~mask]) ** 2
        return out.mean()

    def forward(self, input, target):
        return self._mse_loss(input.view(-1, 18), target, self.ignored_index)