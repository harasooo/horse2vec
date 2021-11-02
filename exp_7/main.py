import copy
import os
from collections import defaultdict, abc
from typing import Dict, List, Set

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset


def flatten(nested_list: List[List[int]]) -> List[int]:
    for el in nested_list:
        if isinstance(el, abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def rank_order(image):
    flat_image = image.ravel()
    sort_order = flat_image.argsort()
    flat_image = flat_image[sort_order]
    sort_rank = np.zeros_like(sort_order)
    is_different = flat_image[:-1] != flat_image[1:]
    np.cumsum(is_different, out=sort_rank[1:])
    original_values = np.zeros((sort_rank[-1] + 1,), image.dtype)
    original_values[0] = flat_image[0]
    original_values[1:] = flat_image[1:][is_different]
    int_image = np.zeros_like(sort_order)
    int_image[sort_order] = sort_rank
    return int_image.reshape(image.shape)


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
            train_dict[id]["emb_id"] = np.ndarray(emb_list)
            train_dict[id]["update_emb_id_before"] = np.ndarray(emb_list)[update_list]
            train_dict[id]["update_emb_id_after"] = np.ndarray(emb_list)[update_list] + 1
    return train_dict


def find_unique_race_list(
    orderd_race_id: List[int], df_order: pd.DataFrame
) -> List[int]:
    horse_set: Set[int] = set()
    unique_race_list = []
    for race_id in reversed(orderd_race_id):
        tmp_horse_set = set(df_order[df_order["race_id"] == race_id]["horse_name_v"])
        if horse_set & tmp_horse_set:
            break
        else:
            horse_set = horse_set | tmp_horse_set
            unique_race_list.append(race_id)
    return unique_race_list


def find_all_unique_race_list(df: pd.DataFrame) -> List[List[int]]:
    df_order = df.sort_values(["date", "race_id"])
    ordered_race_id = df_order["race_id"].unique()
    unique_race_list_list = []
    while len(ordered_race_id) != 0:
        unique_race_list = find_unique_race_list(ordered_race_id, df_order)
        unique_race_list_list.append(unique_race_list)
        ordered_race_id = ordered_race_id[: -len(unique_race_list)]
    return unique_race_list_list

def get_train_sampler(unique_race_list_list: List[List[int]]):


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
            trandform_data = trandform_data.to(torch.int64)
        trandform_data = F.pad(
            trandform_data,
            (0, self.worst_rank - len(trandform_data)),
            "constant",
            self.pad_idx,
        )
        if dtype == "int":
            return trandform_data.to(torch.int64)
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


class CustumBatchHorseDataset(Dataset):
    def __init__(
        self,
        train_dict: Dict[str, np.ndarray],
        sampler: np.ndarray,
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
            trandform_data = trandform_data.to(torch.int64)
        trandform_data = F.pad(
            trandform_data,
            (0, self.worst_rank - len(trandform_data)),
            "constant",
            self.pad_idx,
        )
        if dtype == "int":
            return trandform_data.to(torch.int64)
        else:
            return trandform_data.to(torch.float)

    def __len__(self):
        return len(self.sampler) - 1

    def __getitem__(self, index: int):
        race_id_list = self.sampler[index]
        emb_id_list = []
        covs_list = []
        target_time_list = []
        target_rank_list = []
        mask_list = []
        update_emb_id_before_list = []
        update_emb_id_after_list = []
        for race_id in race_id_list:
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
                [
                    covs,
                    torch.zeros(
                        (self.worst_rank - covs.shape[0]), self.n_added_futures
                    ),
                ]
            )
            mask = self._add_pad_mask(emb_id)

            emb_id_list.append(emb_id)
            covs_list.append(covs)
            target_time_list.append(target_time)
            target_rank_list.append(target_rank)
            mask_list.append(mask)
            update_emb_id_before_list.append(update_emb_id_before)
            update_emb_id_after_list.append(update_emb_id_after)
        return (
            torch.stack(emb_id_list),
            torch.stack(covs_list),
            torch.stack(target_time_list),
            torch.stack(target_rank_list),
            torch.stack(mask_list),
            torch.stack(update_emb_id_before_list),
            torch.stack(update_emb_id_after_list),
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
        self.decoder_1 = nn.Linear((hidden_size + n_futures), hidden_size, bias=True)
        self.decoder_2 = nn.Linear((hidden_size), hidden_size, bias=True)
        self.decoder = nn.Linear((hidden_size), vocab_size, bias=False)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.3)
        self.dropout_2 = nn.Dropout(p=0.3)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states, added_futures):
        hidden_states = self.transform(hidden_states)
        hidden_states = torch.cat((hidden_states, added_futures), 2)
        hidden_states = self.dropout_1(self.relu_1(self.decoder_1(hidden_states)))
        hidden_states = self.dropout_2(self.relu_2(self.decoder_2(hidden_states)))
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
        train_dict: Dict[str, np.ndarray],
        train_sampler: List[int],
        val_sampler: List[int],
        test_sampler: List[int],
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
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.test_sampler = test_sampler
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

        self.test_dataset = HorseDataset(
            self.train_dict,
            self.test_sampler,
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

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
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
        self.time_classifier = BertLMPredictionHead(
            self.d_model, self.layer_eps, 1, n_added_futures
        )
        self.rank_classifier = BertLMPredictionHead(
            self.d_model, self.layer_eps, 1, n_added_futures
        )
        self.time_criterion = CustomMSELoss(self.padding_idx)
        self.rank_criterion = CustomMSELoss(self.padding_idx)

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
        time_out = self.time_classifier(hidden_states, covs)
        rank_out = self.rank_classifier(hidden_states, covs)
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
        time_out, rank_out = self.forward(
            emb_id.squeeze(), covs.squeeze(), mask.squeeze()
        )
        loss_1 = self.time_criterion(time_out.squeeze(), time_target.squeeze())
        loss_2 = self.rank_criterion(rank_out.squeeze(), rank_target.squeeze())
        loss = (loss_1 + self.ranklambda * loss_2) / 2
        self.update_furture_horse_vec(
            update_emb_id_before.squeeze(), update_emb_id_after.squeeze()
        )
        return {
            "loss": loss,
            "rank_batch_preds": rank_out,
            "rank_batch_labels": rank_target.squeeze(),
            "time_batch_preds": time_out,
            "time_batch_labels": time_target.squeeze(),
        }

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
        time_out, rank_out = self.forward(
            emb_id.squeeze(), covs.squeeze(), mask.squeeze()
        )
        loss_1 = self.time_criterion(time_out, time_target.squeeze())
        loss_2 = self.rank_criterion(rank_out, rank_target.squeeze())
        loss = (loss_1 + self.ranklambda * loss_2) / 2
        return {
            "loss": loss,
            "rank_batch_preds": rank_out,
            "rank_batch_labels": rank_target.squeeze(),
            "time_batch_preds": time_out,
            "time_batch_labels": time_target.squeeze(),
        }

    def test_step(self, batch, batch_idx):
        (
            emb_id,
            covs,
            time_target,
            rank_target,
            mask,
            update_emb_id_before,
            update_emb_id_after,
        ) = batch
        time_out, rank_out = self.forward(
            emb_id.squeeze(), covs.squeeze(), mask.squeeze()
        )
        loss_1 = self.time_criterion(time_out, time_target.squeeze())
        loss_2 = self.rank_criterion(rank_out, rank_target.squeeze())
        loss = (loss_1 + self.ranklambda * loss_2) / 2
        return {
            "loss": loss,
            "rank_batch_preds": rank_out,
            "rank_batch_labels": rank_target.squeeze(),
            "time_batch_preds": time_out,
            "time_batch_labels": time_target.squeeze(),
        }

    def training_epoch_end(self, outputs, mode="train"):
        time_epoch_y_hats = torch.cat([x["time_batch_preds"] for x in outputs])
        time_epoch_labels = torch.cat([x["time_batch_labels"] for x in outputs])
        time_epoch_loss = self.time_criterion(time_epoch_y_hats, time_epoch_labels)
        self.log(f"{mode}_time_loss", time_epoch_loss)

        rank_epoch_y_hats = torch.cat([x["rank_batch_preds"] for x in outputs])
        rank_epoch_labels = torch.cat([x["rank_batch_labels"] for x in outputs])
        rank_epoch_loss = self.rank_criterion(rank_epoch_y_hats, rank_epoch_labels)
        self.log(f"{mode}_rank_loss", rank_epoch_loss)

        epoch_loss = (time_epoch_loss + self.ranklambda * rank_epoch_loss) / 2
        self.log(f"{mode}_total_loss", epoch_loss)

        rank_epoch_y_hats_for_auc = rank_order(
            -rank_epoch_y_hats.view(-1).cpu().numpy()
        )
        rank_epoch_labels_for_auc = rank_order(rank_epoch_labels.view(-1).cpu().numpy())
        top1_labels_for_auc = (rank_epoch_labels_for_auc == 0).astype(int)
        top_1_roc_score = roc_auc_score(top1_labels_for_auc, rank_epoch_y_hats_for_auc)
        self.log(f"{mode}_top_1_roc_score", top_1_roc_score)

    def validation_epoch_end(self, outputs, mode="val"):
        time_epoch_y_hats = torch.cat([x["time_batch_preds"] for x in outputs])
        time_epoch_labels = torch.cat([x["time_batch_labels"] for x in outputs])
        time_epoch_loss = self.time_criterion(time_epoch_y_hats, time_epoch_labels)
        self.log(f"{mode}_time_loss", time_epoch_loss)

        rank_epoch_y_hats = torch.cat([x["rank_batch_preds"] for x in outputs])
        rank_epoch_labels = torch.cat([x["rank_batch_labels"] for x in outputs])
        rank_epoch_loss = self.rank_criterion(rank_epoch_y_hats, rank_epoch_labels)
        self.log(f"{mode}_rank_loss", rank_epoch_loss)

        epoch_loss = (time_epoch_loss + self.ranklambda * rank_epoch_loss) / 2
        self.log(f"{mode}_total_loss", epoch_loss)

        rank_epoch_y_hats_for_auc = rank_order(
            -rank_epoch_y_hats.view(-1).cpu().numpy()
        )
        rank_epoch_labels_for_auc = rank_order(rank_epoch_labels.view(-1).cpu().numpy())
        top1_labels_for_auc = (rank_epoch_labels_for_auc == 0).astype(int)
        top_1_roc_score = roc_auc_score(top1_labels_for_auc, rank_epoch_y_hats_for_auc)
        self.log(f"{mode}_top_1_roc_score", top_1_roc_score)

    def test_epoch_end(self, outputs, mode="test"):
        time_epoch_y_hats = torch.cat([x["time_batch_preds"] for x in outputs])
        time_epoch_labels = torch.cat([x["time_batch_labels"] for x in outputs])
        time_epoch_loss = self.time_criterion(time_epoch_y_hats, time_epoch_labels)
        self.log(f"{mode}_time_loss", time_epoch_loss)

        rank_epoch_y_hats = torch.cat([x["rank_batch_preds"] for x in outputs])
        rank_epoch_labels = torch.cat([x["rank_batch_labels"] for x in outputs])
        rank_epoch_loss = self.rank_criterion(rank_epoch_y_hats, rank_epoch_labels)
        self.log(f"{mode}_rank_loss", rank_epoch_loss)

        epoch_loss = (time_epoch_loss + self.ranklambda * rank_epoch_loss) / 91
        self.log(f"{mode}_total_loss", epoch_loss)

        rank_epoch_y_hats_for_auc = rank_order(
            -rank_epoch_y_hats.view(-1).cpu().numpy()
        )
        rank_epoch_labels_for_auc = rank_order(rank_epoch_labels.view(-1).cpu().numpy())
        top1_labels_for_auc = (rank_epoch_labels_for_auc == 0).astype(int)
        top_1_roc_score = roc_auc_score(top1_labels_for_auc, rank_epoch_y_hats_for_auc)
        self.log(f"{mode}_top_1_roc_score", top_1_roc_score)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return [optimizer], [scheduler]


def make_callbacks(min_delta, patience, checkpoint_path, save_top_k, model_name):

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=model_name,
        save_top_k=save_top_k,
        verbose=True,
        monitor="val_total_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_total_loss", min_delta=min_delta, patience=patience, mode="min"
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
        df = pd.read_csv(cfg.path.train_df_path)
        train_dict = pd.read_pickle(cfg.path.train_dic_path)
        own_sampler = find_all_unique_race_list(df)
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
    trainer.test()


if __name__ == "__main__":
    main()
