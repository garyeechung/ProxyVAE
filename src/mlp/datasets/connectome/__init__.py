from typing import List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .utils import get_data_from_df, split_train_val, ADNIConnectomeDataset


def get_adni_dataloaders(df: pd.DataFrame, modality: str, data_dir: str,
                         targets: List[str], num_classes: List[int], one_hot: bool = True,
                         stratified_by: Optional[List[str]] = ["model_type_id"],
                         batch_size: int = 32, batch_per_epoch: int = 100, num_workers: int = 4,
                         include_mappable_site_empty: bool = False,
                         seed: int = 42) -> List[DataLoader]:
    np.random.seed(seed)

    assert modality in ["connectome"], f"modality must be one of ['connectome'], got {modality}"
    AVAILABLE_TARGETS = ["manufacturer_id", "model_type_id", "site"]
    for target in targets:
        assert target in AVAILABLE_TARGETS, f"target must be one of {AVAILABLE_TARGETS}, got {target}"
    df_targets_count = df.groupby(targets).size().reset_index(name="count")
    df_targets_count = df_targets_count.assign(bootstrap=df_targets_count["count"].sum() / df_targets_count["count"])
    df_targets_count.drop(columns=["count"], inplace=True)
    df = df.merge(df_targets_count, on=targets, how="left")

    num_classes = [df[target].max() + 1 for target in targets]

    if stratified_by is not None:
        df_train_val_split = (
            df.loc[~df["holdout"], :]
            .drop_duplicates(subset=["sub"])
            .groupby(stratified_by)["sub"]
            .apply(list)
            .apply(lambda x: split_train_val(x))
            .reset_index()
        )
        df_train_val_split = df_train_val_split.assign(size=lambda x: x["train"].apply(len))
        df_train_val_split["train_ids"] = df_train_val_split["train"].apply(lambda x: df.index[df["sub"].isin(x)].tolist())
        df_train_val_split["valid_ids"] = df_train_val_split["valid"].apply(lambda x: df.index[df["sub"].isin(x)].tolist())
        train_rows = df_train_val_split.apply(lambda x: x["train_ids"], axis=1).sum()
        valid_rows = df_train_val_split.apply(lambda x: x["valid_ids"], axis=1).sum()

    else:
        rest_ids = df.index[~df["holdout"]].tolist()
        rest_ids = np.random.permutation(rest_ids)
        train_size = int(len(rest_ids) * 0.9)
        train_rows = rest_ids[:train_size]
        valid_rows = rest_ids[train_size:]
    holdout_rows = df.index[(df["holdout"]) & (df["manufacturer_id"] != -1)].tolist()

    dataloaders = []
    for i, row in enumerate([train_rows, valid_rows, holdout_rows]):
        df_subset = df.loc[row]
        data = get_data_from_df(df=df_subset, data_dir=data_dir, modality=modality,
                                targets=targets, include_mappable_site_empty=include_mappable_site_empty)
        dataset = ADNIConnectomeDataset(data=data, targets=targets, num_classes=num_classes, one_hot=one_hot)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=(i == 0))
        dataloaders.append(dataloader)

    return dataloaders
