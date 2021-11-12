import torch
import torch.nn as nn


def get_optimizers(model: nn.Module, max_epoch: int, lr: float):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
    return optimizer, scheduler
