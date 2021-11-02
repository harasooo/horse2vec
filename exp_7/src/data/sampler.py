from typing import List, Set

import pandas as pd
from utils import flatten


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


def get_train_sampler_with_leak(unique_race_list_list: List[List[int]]) -> List[int]:
    return list(flatten(unique_race_list_list[:-3]))


def get_val1_sampler(unique_race_list_list: List[List[int]]) -> List[List[int]]:
    return [unique_race_list_list[-3]]


def get_val2_sampler(unique_race_list_list: List[List[int]]) -> List[List[int]]:
    return [unique_race_list_list[-2]]


def get_test_sampler(unique_race_list_list: List[List[int]]) -> List[List[int]]:
    return [unique_race_list_list[-1]]


def get_train_sampler_without_leak(unique_race_list_list: List[List[int]]):
    pass


def get_val_sampler_without_leak(unique_race_list_list: List[List[int]]):
    pass
