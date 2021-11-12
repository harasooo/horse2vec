from typing import Dict, List
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


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


def get_leaked_dataloaders(
    train_dict: Dict[str, np.ndarray],
    train_sampler: List[int],
    val_1_sampler: List[List[int]],
    val_2_sampler: List[List[int]],
    test_sampler: List[List[int]],
    target_time_key: str,
    target_rank_key: str,
    pad_idx: int,
    worst_rank: int,
    n_added_futures: int,
    batch_size: int,
) -> Dict[str, DataLoader]:

    dataloader_dict = {}

    train_dataset = HorseDataset(
        train_dict,
        train_sampler,
        target_time_key,
        target_rank_key,
        pad_idx,
        worst_rank,
        n_added_futures,
    )

    val_1_dataset = CustumBatchHorseDataset(
        train_dict,
        val_1_sampler,
        target_time_key,
        target_rank_key,
        pad_idx,
        worst_rank,
        n_added_futures,
    )

    val_2_dataset = CustumBatchHorseDataset(
        train_dict,
        val_2_sampler,
        target_time_key,
        target_rank_key,
        pad_idx,
        worst_rank,
        n_added_futures,
    )

    test_dataset = CustumBatchHorseDataset(
        train_dict,
        test_sampler,
        target_time_key,
        target_rank_key,
        pad_idx,
        worst_rank,
        n_added_futures,
    )

    dataloader_dict["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
    )

    dataloader_dict["val_1"] = val_1_dataset[0]

    dataloader_dict["val_2"] = val_2_dataset[0]

    dataloader_dict["test"] = test_dataset[0]
    return dataloader_dict
