import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_data_from_df(df: pd.DataFrame, modality: str, data_dir: str, targets: List[str], include_mappable_site_empty=False):

    if not include_mappable_site_empty:
        df = df[~df["site_empty"]]

    assert modality in ["connectome"], f"modality must be one of ['connectome'], got {modality}"

    data = []
    for _, row in df.iterrows():
        item = {"image": os.path.join(data_dir, row[f"path_{modality}"])}
        for target in targets:
            item[target] = row[target]
        for key in ["sub", "ses"]:
            item[key] = row[key]
        data.append(item)

    return data


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


class ADNIConnectomeDataset(Dataset):
    def __init__(self, data: List[dict], targets: List[str], num_classes: List[int], one_hot=True):
        self.data = data
        assert len(targets) == len(num_classes), "tagets and # of classes must be the same"
        self.targets = targets
        self.num_classes = num_classes
        self.one_hot = one_hot

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        datum = self.data[idx]

        x = np.load(datum["image"])
        x = x[np.triu_indices_from(x, k=1)]
        x = np.log1p(x) / np.log1p(100000)
        x = torch.from_numpy(x).float()

        y_all = []
        for target, num_class in zip(self.targets, self.num_classes):
            y = np.array(datum[target])
            if self.one_hot:
                y = np.eye(num_class)[int(y)]
            y = torch.from_numpy(y).float()
            y_all.append(y)

        return x, *y_all
