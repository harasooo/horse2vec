import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Dict


def train_step(
    model: nn.Module,
    train_loader: DataLoader,
    device: str,
    custum_batch: bool,
    time_criterion,
    rank_criterion,
    optimizer,
    scheduler,
    ranklambda,
):
    model.train()
    train_batch_loss = []
    train_time_out_list = []
    train_rank_out_list = []
    train_time_target_list = []
    train_rank_target_list = []
    # train loop
    for (
        emb_id,
        covs,
        time_target,
        rank_target,
        mask,
        update_emb_id_before,
        update_emb_id_after,
    ) in train_loader:

        # trainデータ
        if custum_batch is True:
            emb_id = emb_id.squeeze()
            covs = covs.squeeze()
            time_target = time_target.squeeze()
            rank_target = rank_target.squeeze()
            mask = mask.squeeze()
            update_emb_id_before = update_emb_id_before.squeeze()
            update_emb_id_after = update_emb_id_after.squeeze()
        emb_id = emb_id.to(device)
        covs = covs.to(device)
        time_target = time_target.to(device)
        rank_target = rank_target.to(device)
        mask = mask.to(device)
        update_emb_id_before = update_emb_id_before.to(device)
        update_emb_id_after = update_emb_id_after.to(device)

        # reset grad
        optimizer.zero_grad()

        # forward計算 & Loss計算
        time_out, rank_out = model.forward(emb_id, covs, mask)
        loss_1 = time_criterion(time_out, time_target)
        loss_2 = rank_criterion(rank_out, rank_target)
        loss = (loss_1 + ranklambda * loss_2) / 2

        # backward & step
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.update_furture_horse_vec(update_emb_id_before, update_emb_id_after)

        # 予測値 & loss
        train_time_out_list.append(time_out)
        train_rank_out_list.append(rank_out)
        train_time_target_list.append(time_target)
        train_rank_target_list.append(rank_target)
        train_batch_loss.append(loss.item())

    # 予測値の結合
    oof: Dict[str, np.array] = {}
    oof["time_out"] = torch.cat(train_time_out_list, axis=0).cpu().detach().numpy()
    oof["rank_out"] = torch.cat(train_rank_out_list, axis=0).cpu().detach().numpy()
    oof["time_target"] = (
        torch.cat(train_time_target_list, axis=0).cpu().detach().numpy()
    )
    oof["rank_target"] = (
        torch.cat(train_rank_target_list, axis=0).cpu().detach().numpy()
    )

    return (model, optimizer, scheduler, np.mean(train_batch_loss), oof)


def val_step(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    custum_batch: bool,
    ranklambda,
    time_criterion,
    rank_criterion,
):
    model.eval()
    test_batch_loss = []
    val_batch_loss = []
    val_time_out_list = []
    val_rank_out_list = []
    val_time_target_list = []
    val_rank_target_list = []

    # val loop
    with torch.no_grad():
        for (
            emb_id,
            covs,
            time_target,
            rank_target,
            mask,
            update_emb_id_before,
            update_emb_id_after,
        ) in test_loader:

            # valデータ
            if custum_batch is True:
                emb_id = emb_id.squeeze()
                covs = covs.squeeze()
                time_target = time_target.squeeze()
                rank_target = rank_target.squeeze()
                mask = mask.squeeze()
                update_emb_id_before = update_emb_id_before.squeeze()
                update_emb_id_after = update_emb_id_after.squeeze()
            emb_id = emb_id.to(device)
            covs = covs.to(device)
            time_target = time_target.to(device)
            rank_target = rank_target.to(device)
            mask = mask.to(device)
            update_emb_id_before = update_emb_id_before.to(device)
            update_emb_id_after = update_emb_id_after.to(device)

            # forward計算 & Loss計算
            time_out, rank_out = model.forward(emb_id, covs, mask)

            # 予測値 & loss
            loss_1 = time_criterion(time_out, time_target)
            loss_2 = rank_criterion(rank_out, rank_target)
            loss = (loss_1 + ranklambda * loss_2) / 2
            test_batch_loss.append(loss.item())

            # 予測値 & loss
            val_time_out_list.append(time_out)
            val_rank_out_list.append(rank_out)
            val_time_target_list.append(time_target)
            val_rank_target_list.append(rank_target)
            val_batch_loss.append(loss.item())

    print(val_rank_out_list)
    print(val_time_out_list)
    # 予測値の結合
    oof: Dict[str, np.array] = {}
    oof["time_out"] = torch.cat(val_time_out_list, axis=0).cpu().detach().numpy()
    oof["rank_out"] = torch.cat(val_rank_out_list, axis=0).cpu().detach().numpy()
    oof["time_target"] = torch.cat(val_time_target_list, axis=0).cpu().detach().numpy()
    oof["rank_target"] = torch.cat(val_rank_target_list, axis=0).cpu().detach().numpy()

    return (
        model,
        np.mean(test_batch_loss),
        oof,
    )
