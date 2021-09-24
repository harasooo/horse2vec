import copy
import os
from collections import defaultdict
from typing import Dict

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset


def make_time_horses_emb_table(df: pd.DataFrame):
    time_hores_emb_table = defaultdict(lambda: defaultdict(int))
    emb_id = 0
    for ml in (
        df[["horse_name_v", "race_id"]]
        .groupby(["horse_name_v", "race_id"])
        .count()
        .index
    ):
        hores_id = ml[0]
        race_id = ml[1]
        time_hores_emb_table[hores_id][race_id] = emb_id
        emb_id += 1
    return time_hores_emb_table


def make_train_dict(df: pd.DataFrame, time_hores_emb_table: dict) -> Dict:
    train_dict = {}
    df["rank"] = df["rank"].map(lambda x: str(x).replace("(降)", "").replace("失", "18"))
    for id in df["race_id"].unique():
        train_dict[id] = {
            "horses": df[df["race_id"] == id]["horse_name_v"].values,
            "time": df[df["race_id"] == id]["time_s"].values,
            "covatiates": df[df["race_id"] == id].iloc[:, 7:].values,
            "rank": df[df["race_id"] == id]["rank"].values,
        }
        emb_list = []
        update_list = []
        for hores_id in train_dict[id]["horses"]:
            emb_id = time_hores_emb_table[hores_id][id]
            emb_list.append(emb_id)
            if (
                len(
                    [
                        v
                        for k, v in time_hores_emb_table[hores_id].items()
                        if v == (emb_id + 1)
                    ]
                )
                > 0
            ):
                update_list.append(True)
            else:
                update_list.append(False)
            train_dict[id]["emb_id"] = np.array(emb_list)
            train_dict[id]["update_emb_id_before"] = np.array(emb_list)[update_list]
            train_dict[id]["update_emb_id_after"] = np.array(emb_list)[update_list] + 1
    return train_dict


class HorseDataset(Dataset):
    def __init__(
        self,
        train_dict: Dict[str, np.array],
        sampler: np.array,
        target_time_key: str,
        target_rank_key: str,
        pad_idx: int,
        worst_rank: int,
        n_added_futures: int,
    ):
        self.train_dict = train_dict
        self.sampler = sampler
        self.target_time_key = target_time_key
        self.target_rank_key = target_rank_key
        self.pad_idx = pad_idx
        self.n_added_futures = n_added_futures
        self.worst_rank = worst_rank

    def _add_pad_mask(self, data: torch.Tensor):
        return torch.Tensor([False if i != self.pad_idx else True for i in data]).to(
            torch.bool
        )

    def _to_pad_torch_type(self, data, key, dtype):
        trandform_data = torch.Tensor(data[key])
        if dtype == "int":
            trandform_data = torch.Tensor(data[key]).to(torch.int64)
        trandform_data = F.pad(
            trandform_data,
            (0, self.worst_rank - len(trandform_data)),
            "constant",
            self.pad_idx,
        )
        if dtype == "int":
            return trandform_data.to(torch.int)
        else:
            return trandform_data.to(torch.float)

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index: int):
        race_id = self.sampler[index]
        data = self.train_dict[race_id]
        emb_id = self._to_pad_torch_type(data, "emb_id", "int")
        target_time = self._to_pad_torch_type(data, self.target_time_key, "float")
        target_rank = self._to_pad_torch_type(data, self.target_rank_key, "int")
        update_emb_id_before = self._to_pad_torch_type(
            data, "update_emb_id_before", "int"
        )
        update_emb_id_after = self._to_pad_torch_type(
            data, "update_emb_id_after", "int"
        )
        covs = torch.Tensor(data["covatiates"])
        covs = torch.cat(
            [covs, torch.zeros((self.worst_rank - covs.shape[0]), self.n_added_futures)]
        )
        mask = self._add_pad_mask(emb_id)
        return (
            emb_id,
            covs,
            target_time,
            target_rank,
            mask,
            update_emb_id_before,
            update_emb_id_after,
        )


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, layer_norm_eps, vocab_size, n_futures):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, layer_norm_eps)
        self.decoder = nn.Linear((hidden_size + n_futures), vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states, added_futures):
        hidden_states = self.transform(hidden_states)
        hidden_states = torch.cat((hidden_states, added_futures), 2)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


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


class CreateDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dict: Dict[str, np.array],
        sampler: np.array,
        val_num: int,
        target_time_key: str,
        target_rank_key: str,
        pad_idx: int,
        worst_rank: int,
        n_added_futures: int,
        batch_size: int,
    ):
        super().__init__()
        self.train_dict = train_dict
        self.train_sampler = sampler[:-val_num]
        self.val_sampler = sampler[-val_num:]
        self.target_time_key = target_time_key
        self.target_rank_key = target_rank_key
        self.pad_idx = pad_idx
        self.batch_size = batch_size
        self.worst_rank = worst_rank
        self.n_added_futures = n_added_futures

    def setup(self):
        self.train_dataset = HorseDataset(
            self.train_dict,
            self.train_sampler,
            self.target_time_key,
            self.target_rank_key,
            self.pad_idx,
            self.worst_rank,
            self.n_added_futures,
        )

        self.val_dataset = HorseDataset(
            self.train_dict,
            self.val_sampler,
            self.target_time_key,
            self.target_rank_key,
            self.pad_idx,
            self.worst_rank,
            self.n_added_futures,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
        )


class CustumBert(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        learning_rate: float,
        padding_idx: int,
        worst_rank: int,
        layer_eps: float,
        num_heads: int,
        n_times: int,
        n_added_futures: int,
        batch_size: int,
        dropout: float,
        ranklambda: float,
    ):
        super().__init__()

        self.padding_idx = padding_idx
        self.worst_rank = worst_rank
        self.batch_size = batch_size
        self.d_model = d_model
        self.layer_eps = layer_eps
        self.lr = learning_rate
        self.dropout = dropout
        self.n_times = n_times - 1
        self.ranklambda = ranklambda

        self.emb = nn.Embedding(self.padding_idx + 1, self.d_model, self.padding_idx)
        self.attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.d_model,
                    num_heads=num_heads,
                    dropout=self.dropout,
                    batch_first=True,
                )
                for _ in range(self.n_times)
            ]
        )
        self.lns = nn.ModuleList(
            [nn.LayerNorm(self.d_model, self.layer_eps) for _ in range(self.n_times)]
        )

        self.attn_last = nn.MultiheadAttention(
            self.d_model, num_heads=num_heads, dropout=self.dropout, batch_first=True
        )
        self.classifier = BertLMPredictionHead(
            self.d_model, self.layer_eps, 1, n_added_futures
        )

        self.time_criterion = CustomMSELoss(self.padding_idx)
        self.rank_criterion = nn.CrossEntropyLoss(ignore_index=self.padding_idx)

        for param in self.parameters():
            param.requires_grad = True

    def update_furture_horse_vec(self, update_emb_id_before, update_emb_id_after):
        after = update_emb_id_after.view(-1).to(torch.int64)
        before = update_emb_id_before.view(-1).to(torch.int64)
        self.emb.weight.data[after] = self.emb.weight.data[before].clone()

    def forward(self, horses, covs, pad_mask):
        atten_inputs = self.emb(horses)
        for i in range(self.n_times):
            hidden_states = self.attns[i](
                atten_inputs, atten_inputs, atten_inputs, key_padding_mask=pad_mask
            )[0]
            hidden_states = self.lns[i](hidden_states)
        hidden_states = self.attn_last(
            atten_inputs, atten_inputs, atten_inputs, key_padding_mask=pad_mask
        )[0]
        time_out = self.classifier(hidden_states, covs)
        rank_out = self.classifier(hidden_states, covs)
        return time_out, rank_out

    def training_step(self, batch, batch_idx):
        (
            emb_id,
            covs,
            time_target,
            rank_target,
            mask,
            update_emb_id_before,
            update_emb_id_after,
        ) = batch
        time_out, rank_out = self.forward(emb_id, covs, mask)
        loss_1 = self.time_criterion(time_out, time_target)
        loss_2 = self.rank_criterion(rank_out, rank_target)
        loss = loss_1 + self.ranklambda * loss_2
        self.update_furture_horse_vec(update_emb_id_before, update_emb_id_after)
        return {"loss": loss, "batch_preds": rank_out, "batch_labels": rank_target}

    def validation_step(self, batch, batch_idx):
        (
            emb_id,
            covs,
            time_target,
            rank_target,
            mask,
            update_emb_id_before,
            update_emb_id_after,
        ) = batch
        time_out, rank_out = self.forward(emb_id, covs, mask)
        loss_1 = self.time_criterion(time_out, time_target)
        loss_2 = self.rank_criterion(rank_out, rank_target)
        loss = loss_1 + self.ranklambda * loss_2
        self.update_furture_horse_vec(update_emb_id_before, update_emb_id_after)
        return {"loss": loss, "batch_preds": rank_out, "batch_labels": rank_target}

    def training_epoch_end(self, outputs, mode="train"):
        epoch_y_hats = torch.cat([x["batch_preds"] for x in outputs])
        epoch_labels = torch.cat([x["batch_labels"] for x in outputs])
        epoch_loss = self.criterion(epoch_y_hats, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss)

        _, epoch_preds = torch.max(epoch_y_hats, 1)
        epoch_accuracy = accuracy(epoch_preds, epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy)

    def validation_epoch_end(self, outputs, mode="val"):
        epoch_y_hats = torch.cat([x["batch_preds"] for x in outputs])
        epoch_labels = torch.cat([x["batch_labels"] for x in outputs])
        epoch_loss = self.criterion(epoch_y_hats, epoch_labels)
        self.log(f"{mode}_loss", epoch_loss)

        _, epoch_preds = torch.max(epoch_y_hats, 1)
        epoch_accuracy = accuracy(epoch_preds, epoch_labels)
        self.log(f"{mode}_accuracy", epoch_accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def make_callbacks(min_delta, patience, checkpoint_path, save_top_k, model_name):

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=model_name,
        save_top_k=save_top_k,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=min_delta, patience=patience, mode="min"
    )

    return [early_stop_callback, checkpoint_callback]


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    cwd = hydra.utils.get_original_cwd()
    print(f"cwd:{cwd}")
    tags = copy.deepcopy(cfg.wandb.tags)
    wandb_logger = WandbLogger(
        name=("exp_" + str(cfg.wandb.exp_name)),
        project=cfg.wandb.project,
        tags=tags,
        log_model=True,
    )
    checkpoint_path = os.path.join(
        wandb_logger.experiment.dir, cfg.path.checkpoint_path
    )
    wandb_logger.log_hyperparams(cfg)

    if cfg.path.read_trin_dict is True:
        train_dict = pd.read_pickle(cfg.path.train_dic_path)
        own_sampler = pd.read_pickle(cfg.path.own_sampler_path)
    else:
        df = pd.read_csv(cfg.path.train_df_path)
        time_horses_emb_table = make_time_horses_emb_table(df)
        train_dict = make_train_dict(df, time_horses_emb_table)
        own_sampler = (
            df[["race_id", "date"]]
            .groupby(["race_id"], as_index=False)
            .max()
            .sort_values("date")["race_id"]
            .values
        )

    data_module = CreateDataModule(
        train_dict=train_dict,
        sampler=own_sampler,
        val_num=cfg.training.val_nun,
        target_time_key=cfg.training.target_time_key,
        target_rank_key=cfg.training.target_rank_key,
        pad_idx=cfg.model.pad_idx,
        worst_rank=cfg.model.worst_rank,
        n_added_futures=cfg.model.n_added_futures,
        batch_size=cfg.training.batch_size,
    )
    data_module.setup()

    call_backs = make_callbacks(
        min_delta=cfg.callbacks.patience_min_delta,
        patience=cfg.callbacks.patience,
        checkpoint_path=checkpoint_path,
        save_top_k=cfg.callbacks.save_top_k,
        model_name="exp_" + str(cfg.wandb.exp_name),
    )
    model = CustumBert(
        d_model=cfg.model.d_model,
        learning_rate=cfg.training.learning_rate,
        padding_idx=cfg.model.pad_idx,
        worst_rank=cfg.model.worst_rank,
        layer_eps=cfg.training.layer_eps,
        num_heads=cfg.model.num_heads,
        n_times=cfg.model.n_times,
        dropout=cfg.model.dropout,
        n_added_futures=cfg.model.n_added_futures,
        batch_size=cfg.training.batch_size,
        ranklambda=cfg.training.ranklambda,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        gpus=1,
        progress_bar_refresh_rate=30,
        callbacks=call_backs,
        logger=wandb_logger,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
