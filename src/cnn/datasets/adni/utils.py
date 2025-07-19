import os
from typing import List

from monai.data import Dataset, PersistentDataset, CacheDataset
import numpy as np
import pandas as pd


def get_data_from_df(df: pd.DataFrame, modality: str, data_dir: str, targets: List[str], include_mappable_site_empty=False):

    if not include_mappable_site_empty:
        df = df[~df["site_empty"]]

    assert modality in ["fa", "t1"], f"modality must be one of ['fa', 't1'], got {modality}"

    data = []
    for _, row in df.iterrows():
        item = {"image": os.path.join(data_dir, row[f"path_{modality}"])}
        for target in targets:
            item[target] = row[target]
        for key in ["sub", "ses"]:
            item[key] = row[key]
        data.append(item)

    return data


def get_dataset_from_data(data: List[dict], cache_transforms, cache_type=None, cache_dir=None):
    """
    Create a dataset from a list of data dictionaries and apply cache transforms.
    """
    if cache_type is None:
        return Dataset(data=data, transform=cache_transforms)
    elif cache_type == "persistent":
        return PersistentDataset(data=data, transform=cache_transforms, cache_dir=cache_dir)
    elif cache_type == "cache":
        return CacheDataset(data=data, transform=cache_transforms, cache_rate=1.0)


def split_train_val(list_: List) -> pd.Series:
    """
    Split a list into training and validation sets.
    """
    seed = 42
    val_size = 0.1
    np.random.seed(seed)
    shuffled_list = np.random.permutation(list_)
    split_index = int(len(shuffled_list) * (1 - val_size))
    train_list = shuffled_list[:split_index]
    valid_list = shuffled_list[split_index:]
    return pd.Series({'train': train_list, 'valid': valid_list})
