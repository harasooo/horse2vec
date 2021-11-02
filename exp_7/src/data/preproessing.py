from collections import defaultdict
from typing import Dict

import numpy as np
import pandas as pd


def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        df["distance"].apply(lambda x: False if ("障" in x) or (len(x) != 7) else True)
    ]
    df = df[~df["time"].isna()]
    return df


def to_numerical(df: pd.DataFrame, preprocessed_df_path) -> pd.DataFrame:
    df["turf"] = df["distance"].apply(lambda x: x[0])
    df["turn"] = df["distance"].apply(lambda x: x[1])
    df["meter"] = df["distance"].apply(lambda x: x[2:6])
    df["sex"] = df["age"].apply(lambda x: x[0])
    df["pure_age"] = df["age"].apply(lambda x: x[1:])
    df["weather"] = df["weather"].apply(lambda x: x.replace("天候 : ", ""))
    df["state"] = df["turf_state"].apply(
        lambda x: x.replace("ダート : ", "").replace("芝 : ", "")
    )
    df["time_s"] = df["time"].apply(lambda x: int(x[0])) * 60 + df["time"].apply(
        lambda x: float(x[2:])
    )
    df["pure_weight"] = df["weight"].apply(lambda x: x[:3])
    df["delta_weight"] = df["weight"].apply(
        lambda x: int(x[3:].replace("(+", "").replace("(", "").replace(")", ""))
    )

    JOCKY_DIC = {}
    for v, j_name in enumerate(df["jockey"].unique()):
        JOCKY_DIC[j_name] = v

    OWNER_DIC = {}
    for v, j_name in enumerate(df["owner"].unique()):
        OWNER_DIC[j_name] = v

    TRAINER_DIC = {}
    for v, t_name in enumerate(df["Trainer"].unique()):
        TRAINER_DIC[t_name] = v

    OWNER_DIC = {}
    for v, o_name in enumerate(df["owner"].unique()):
        OWNER_DIC[o_name] = v

    HORSE_DIC = {}
    for v, h_name in enumerate(df["horse_name"].unique()):
        HORSE_DIC[h_name] = v

    RACETRACK_DIC = {}
    for v, h_name in enumerate(df["race_track"].unique()):
        RACETRACK_DIC[h_name] = v

    STATE_DIC = {}
    for v, s_name in enumerate(df["state"].unique()):
        STATE_DIC[s_name] = v

    WEATHER_DIC = {}
    for v, w_name in enumerate(df["weather"].unique()):
        WEATHER_DIC[w_name] = v

    TURF_DIC = {}
    for v, s_name in enumerate(df["turf"].unique()):
        TURF_DIC[s_name] = v

    TURN_DIC = {}
    for v, s_name in enumerate(df["turn"].unique()):
        TURN_DIC[s_name] = v

    SEX_DIC = {}
    for v, s_name in enumerate(df["sex"].unique()):
        SEX_DIC[s_name] = v

    df["jockey_v"] = df["jockey"].apply(lambda x: JOCKY_DIC[x])
    df["Trainer_v"] = df["Trainer"].apply(lambda x: TRAINER_DIC[x])
    df["owner_v"] = df["owner"].apply(lambda x: OWNER_DIC[x])
    df["horse_name_v"] = df["horse_name"].apply(lambda x: HORSE_DIC[x])
    df["race_track_v"] = df["race_track"].apply(lambda x: RACETRACK_DIC[x])
    df["state_v"] = df["state"].apply(lambda x: STATE_DIC[x])
    df["weather_v"] = df["weather"].apply(lambda x: WEATHER_DIC[x])
    df["turf_v"] = df["turf"].apply(lambda x: TURF_DIC[x])
    df["turn_v"] = df["turn"].apply(lambda x: TURN_DIC[x])
    df["sex_v"] = df["sex"].apply(lambda x: SEX_DIC[x])

    df[
        [
            "race_id",
            "date",
            "horse_name_v",
            "rank",
            "time_s",
            "halon_time",
            "post_position",
            "impost",
            "halon_time",
            "popular_rank",
            "win_rate",
            "pure_age",
            "meter",
            "jockey_v",
            "Trainer_v",
            "owner_v",
            "race_track_v",
            "state_v",
            "weather_v",
            "turf_v",
            "turn_v",
            "sex_v",
            "pure_weight",
            "delta_weight",
            "popular_rank",
            "",
        ]
    ].reset_index(drop=True).to_csv(preprocessed_df_path, index=False)

    return df


def make_time_horses_emb_table(df: pd.DataFrame):
    time_hores_emb_table: defaultdict = defaultdict(lambda: defaultdict(int))
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
            train_dict[id]["update_emb_id_after"] = (
                np.ndarray(emb_list)[update_list] + 1
            )
    return train_dict


def raw_to_traindict(
    raw_df_path: str, preprocessed_df_path: str, train_dict_path: str
) -> Dict:
    df = pd.read_csv(raw_df_path)
    df = remove_noise(df)
    df = to_numerical(df, preprocessed_df_path)
    time_horses_emb_table = make_time_horses_emb_table(df)
    train_dict = make_train_dict(df, time_horses_emb_table)
    return train_dict
