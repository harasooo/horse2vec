import copy

import hydra
import wandb
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from data.preproessing import raw_to_traindict
from data.sampler import (
    find_all_unique_race_list,
    get_train_sampler_with_leak,
    get_val1_sampler,
    get_val2_sampler,
    get_test_sampler,
)
from data.dataset import get_leaked_dataloaders
from wandb.utils import transform_log_hyperparams
from net.step1_net import CustumBert
from net.optimizers import get_optimizers
from train.step import train_step, val_step


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    wandb.init(config=(transform_log_hyperparams(cfg)))

    # データの前処理
    raw_df = pd.read_csv(cfg.path.train_df_path)
    train_dict = raw_to_traindict(raw_df, cfg.path.train_df_path)
    all_unique_race_list = find_all_unique_race_list(raw_df)
    # サンプラーの作成
    train_sampler = get_train_sampler_with_leak(all_unique_race_list)
    val_1_sampler = get_val1_sampler(all_unique_race_list)
    val_2_sampler = get_val2_sampler(all_unique_race_list)
    test_sampler = get_test_sampler(all_unique_race_list)

    # dataloaderを作成
    dataloader_dict = get_leaked_dataloaders(
        train_dict=train_dict,
        train_sampler=train_sampler,
        val_1_sampler=val_1_sampler,
        val_2_sampler=val_2_sampler,
        test_sampler=test_sampler,
        target_time_key=cfg.training.target_time_key,
        target_rank_key=cfg.training.target_rank_key,
        pad_idx=cfg.model.pad_idx,
        worst_rank=cfg.model.worst_rank,
        n_added_futures=cfg.model.n_added_futures,
        batch_size=cfg.model.train_batch_size,
    )

    # modelの作成
    model = CustumBert(
        d_model=cfg.model.d_model,
        learning_rate=cfg.model.learning_rate,
        padding_idx=cfg.model.pad_idx,
        worst_rank=cfg.model.worst_rank,
        layer_eps=cfg.model.layer_eps,
        num_heads=cfg.model.num_heads,
        n_times=cfg.model.n_times,
        n_added_futures=cfg.model.n_added_futures,
        dropout=cfg.model.dropout,
    )

    optimizer, scheduler = get_optimizers(
        model, cfg.trainning.n_epochs, cfg.model.learning_rate
    )

    for epoch in tqdm(range(cfg.trainning.n_epochs)):
        (
            model,
            optimizer,
            scheduler,
            train_batch_loss,
            train_time_out,
            train_rank_out,
            train_time_target,
            train_rank_target,
        ) = train_step(
            model=model,
            train_loader=dataloader_dict["train"],
            device=cfg.training.device,
            custum_batch_train=cfg.training.custum_batch_train,
            time_criterion=cfg.training.target_time_key,
            rank_criterion=cfg.training.target_rank_key,
            optimizer=optimizer,
            scheduler=scheduler,
            ranklambda=cfg.training.ranklambda,
        )
