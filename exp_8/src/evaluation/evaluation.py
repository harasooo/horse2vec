from typing import Dict, Tuple, Callable

import numpy as np
from sklearn.metrics import ndcg_score, roc_auc_score

from evaluation.utils import rank_order


def top_1_auc(oof: Dict[str, np.array]) -> Tuple[str, float]:
    rank_out = rank_order(oof["rank_out"]).reshape(-1)
    rank_target = oof["rank_target"].astype(int)
    top1_labels_for_auc = (rank_target == 1).astype(int).reshape(-1)
    top_1_auc_score = roc_auc_score(top1_labels_for_auc, rank_out)
    return "top_1_auc", top_1_auc_score


def top3_ndcg_score(oof: Dict[str, np.array]) -> Tuple[str, float]:
    rank_out = rank_order(oof["rank_out"]) + 1
    rank_target = oof["rank_target"].astype(int)
    top_3_ndcg_score = ndcg_score(1 / rank_target, 1 / rank_out, k=3)
    return "top_3_ndcg", top_3_ndcg_score


def top_1_auc_from_time(oof: Dict[str, np.array]) -> Tuple[str, float]:
    rank_out = rank_order(oof["time_out"]).reshape(-1)
    rank_target = oof["rank_target"].astype(int)
    top1_labels_for_auc = (rank_target == 1).astype(int).reshape(-1)
    top_1_auc_score = roc_auc_score(top1_labels_for_auc, rank_out)
    return "top_1_auc_from_time", top_1_auc_score


def top3_ndcg_score_from_time(oof: Dict[str, np.array]) -> Tuple[str, float]:
    rank_out = rank_order(oof["time_out"]) + 1
    rank_target = oof["rank_target"].astype(int)
    top_3_ndcg_score = ndcg_score(1 / rank_target, 1 / rank_out, k=3)
    return "top_3_ndcg_from_time", top_3_ndcg_score


def make_metrics_func_list():
    metrics_func_list = [
        top_1_auc,
        top_1_auc_from_time,
        top3_ndcg_score,
        top3_ndcg_score_from_time,
    ]
    return metrics_func_list
