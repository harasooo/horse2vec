from torch.utils.data import DataLoader
from typing import Callable
import torch.nn as nn


class Trainer:
    def __init__(self):
        self.train_loader: DataLoader
        self.val_1_loader: DataLoader
        self.val_2_loader: DataLoader
        self.test_loaer: DataLoader
        self.model: nn.Module
        self.trains_setp: Callable
        self.val_1_setp: Callable
        self.val_2_step: Callable
        self.test_step: Callable
        self.max_epoch: int
        self.device: str
        self.optimizer
        self.scheduler
        self.logger
        self.val_1_epoch

    def fit():
        pass