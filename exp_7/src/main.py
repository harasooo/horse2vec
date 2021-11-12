import os

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

import wandb
from data.dataset import get_leaked_dataloaders
from data.preproessing import raw_to_traindict
from data.sampler import (
    find_all_unique_race_list,
    get_test_sampler,
    get_train_sampler_with_leak,
    get_val1_sampler,
    get_val2_sampler,
)
from evaluation.evaluation import make_metrics_func_list
from net.callbacks import EarlyStopper
from net.optimizers import get_optimizers
from net.step1_net import CustumBert
from train.step import train_step, val_step
from logger.utils import transform_log_hyperparams


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    wandb.init(config=(transform_log_hyperparams(cfg)))

    # データの前処理
    raw_df = pd.read_csv(cfg.path.train_df_path)
    if cfg.path.read_train_dict:
        train_dict = pd.read_pickle(cfg.path.train_dic_path)
    else:
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

    # optimizerの設定
    optimizer, scheduler = get_optimizers(
        model, cfg.trainning.n_epochs, cfg.model.learning_rate
    )

    # step1_1(train)
    train_es = EarlyStopper(max_patient=cfg.callbacks.patience)

    # create metrics
    metrics_func_list = make_metrics_func_list()

    for epoch in tqdm(range(cfg.trainning.n_epochs)):
        (model, optimizer, scheduler, train_batch_loss, train_eva_data,) = train_step(
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
        (model, val_batch_loss, val_eva_data,) = val_step(
            model=model,
            train_loader=dataloader_dict["val_1"],
            device=cfg.training.device,
            custum_batch_val=cfg.training.custum_batch_val,
            time_criterion=cfg.training.target_time_key,
            rank_criterion=cfg.training.target_rank_key,
            ranklambda=cfg.training.ranklambda,
        )

        wandb.log({"step_1_1_train_loss": train_batch_loss})
        wandb.log({"step_1_1_val_loss": val_batch_loss})

        for metrics_func in metrics_func_list:
            met_name, met_v = metrics_func(train_eva_data)
            wandb.log({f"step_1_1_train_{met_name}": met_v})
            met_name, met_v = metrics_func(val_eva_data)
            wandb.log({f"step_1_1_val_{met_name}": met_v})

        train_es.update(val_batch_loss)
        if train_es.patience == 0:
            torch.save(
                model.state_dict(), os.path.join(wandb.run.dir, "train_model.h5")
            )
        if train_es.finish:
            break

    # step1_2(to decide num epoch for test step)
    val_es = EarlyStopper(max_patient=cfg.callbacks.patience)
    BEST_EPOCH = 0
    while True:
        (model, optimizer, scheduler, train_batch_loss, train_eva_data,) = train_step(
            model=model,
            train_loader=dataloader_dict["val_1"],
            device=cfg.training.device,
            custum_batch_train=cfg.training.custum_batch_train,
            time_criterion=cfg.training.target_time_key,
            rank_criterion=cfg.training.target_rank_key,
            optimizer=optimizer,
            scheduler=scheduler,
            ranklambda=cfg.training.ranklambda,
        )
        (model, val_batch_loss, val_eva_data,) = val_step(
            model=model,
            train_loader=dataloader_dict["val_2"],
            device=cfg.training.device,
            custum_batch_val=cfg.training.custum_batch_val,
            time_criterion=cfg.training.target_time_key,
            rank_criterion=cfg.training.target_rank_key,
            ranklambda=cfg.training.ranklambda,
        )

        BEST_EPOCH += 1

        wandb.log({"step_1_2_val_1_loss": train_batch_loss})
        wandb.log({"step_1_2_val_2_loss": val_batch_loss})

        for metrics_func in metrics_func_list:
            met_name, met_v = metrics_func(train_eva_data)
            wandb.log({f"step_1_2_train_{met_name}": met_v})
            met_name, met_v = metrics_func(val_eva_data)
            wandb.log({f"step_1_2_val_{met_name}": met_v})

        val_es.update(val_batch_loss)
        if val_es.patience == 0:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "val_model.h5"))
        if val_es.finish:
            break

        # step1_3(test)
    for epoch in tqdm(range(BEST_EPOCH)):
        (model, optimizer, scheduler, train_batch_loss, train_eva_data,) = train_step(
            model=model,
            train_loader=dataloader_dict["val_2"],
            device=cfg.training.device,
            custum_batch_train=cfg.training.custum_batch_train,
            time_criterion=cfg.training.target_time_key,
            rank_criterion=cfg.training.target_rank_key,
            optimizer=optimizer,
            scheduler=scheduler,
            ranklambda=cfg.training.ranklambda,
        )
        (model, val_batch_loss, val_eva_data,) = val_step(
            model=model,
            train_loader=dataloader_dict["test"],
            device=cfg.training.device,
            custum_batch_val=cfg.training.custum_batch_val,
            time_criterion=cfg.training.target_time_key,
            rank_criterion=cfg.training.target_rank_key,
            ranklambda=cfg.training.ranklambda,
        )

        wandb.log({"step_1_3_val_2_loss": train_batch_loss})
        wandb.log({"step_1_3_test_loss": val_batch_loss})

        for metrics_func in metrics_func_list:
            met_name, met_v = metrics_func(train_eva_data)
            wandb.log({f"step_1_3_train_{met_name}": met_v})
            met_name, met_v = metrics_func(val_eva_data)
            wandb.log({f"step_1_3_val_{met_name}": met_v})

    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "test_model.h5"))


if __name__ == "__main__":
    main()
