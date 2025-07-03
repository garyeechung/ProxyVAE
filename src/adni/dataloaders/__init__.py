import os
from typing import List, Optional

from monai.data import Dataset, DataLoader
from monai.transforms import Compose, ToTensord, Transposed
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler

from .transforms import get_cache_transforms
from .transforms import OneHotEncoded, GetCenterSliced, GetRandomSliced
from .utils import get_dataset_from_data, get_data_from_df, split_train_val


def get_adni_dataloaders(df: pd.DataFrame, targets: List[str], data_dir: str,
                         stratified_by: Optional[List[str]] = ["model_type_id"],
                         bootstrap: bool = True, batch_size: int = 32,
                         batch_per_epoch: int = 100, num_workers: int = 4,
                         include_mappable_site_empty: bool = False,
                         seed: int = 42, target_size_in_mm: float = 256.0,
                         target_zooms: Optional[List[float]] = [1.0, 1.0, 1.0],
                         skip_zooming_depth: bool = False,
                         cache_type: Optional[str] = "persistent"):
    np.random.seed(seed)
    cache_dir = os.path.join(data_dir, "cache")

    AVAILABLE_TARGETS = ["manufacturer_id", "model_type_id", "site"]
    for target in targets:
        assert target in AVAILABLE_TARGETS, f"target must be one of {AVAILABLE_TARGETS}, got {target}"
    df_targets_count = df.groupby(targets).size().reset_index(name="count")
    df_targets_count = df_targets_count.assign(bootstrap=df_targets_count["count"].sum() / df_targets_count["count"])
    df_targets_count.drop(columns=["count"], inplace=True)
    df = df.merge(df_targets_count, on=targets, how="left")

    AVAILABLE_CACHE_TYPES = ["persistent", "cache"]
    assert cache_type in AVAILABLE_CACHE_TYPES, \
        f"cache_type must be one of {AVAILABLE_CACHE_TYPES}, got {cache_type}"

    num_classes = [df[target].max() + 1 for target in targets]

    spatial_size = [int(target_size_in_mm / target_zooms[0]),
                    int(target_size_in_mm / target_zooms[1])]

    cache_transforms = get_cache_transforms(image_keys=["image"],
                                            label_keys=targets,
                                            target_zooms=target_zooms,
                                            spatial_size=spatial_size,
                                            skip_zooming_depth=skip_zooming_depth,
                                            slice_range_from_center=0.1)
    train_transforms = Compose([
        GetRandomSliced(keys=["image"]),
        Transposed(keys=["image"], indices=(2, 1, 0)),
        OneHotEncoded(keys=targets, num_classes=num_classes),
        ToTensord(keys=["image", *targets])
    ])
    valid_transforms = Compose([
        GetCenterSliced(keys=["image"]),
        Transposed(keys=["image"], indices=(2, 1, 0)),
        OneHotEncoded(keys=targets, num_classes=num_classes),
        ToTensord(keys=["image", *targets])
    ])
    unknown_transforms = Compose([
        GetCenterSliced(keys=["image"]),
        Transposed(keys=["image"], indices=(2, 1, 0)),
        ToTensord(keys=["image"])
    ])

    dataloaders = []

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

    # Training dataset
    df_subset = df.loc[train_rows]
    data = get_data_from_df(df_subset, data_dir, targets, include_mappable_site_empty)
    cache_dataset = get_dataset_from_data(data, cache_transforms, cache_type=cache_type,
                                          cache_dir=cache_dir)
    dataset = Dataset(data=cache_dataset, transform=train_transforms)
    if bootstrap:
        weights = df_subset["bootstrap"].values
        sampler = WeightedRandomSampler(weights=weights, replacement=True,
                                        num_samples=batch_per_epoch * batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers)
    dataloaders.append(dataloader)

    # Validation dataset
    df_subset = df.loc[valid_rows]
    data = get_data_from_df(df=df_subset, data_dir=data_dir, targets=targets,
                            include_mappable_site_empty=include_mappable_site_empty)
    cache_dataset = get_dataset_from_data(data, cache_transforms, cache_type=cache_type,
                                          cache_dir=cache_dir)
    dataset = Dataset(data=cache_dataset, transform=valid_transforms)
    if bootstrap:
        weights = df_subset["bootstrap"].values
        sampler = WeightedRandomSampler(weights=weights, replacement=True,
                                        num_samples=len(valid_rows))
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)
    dataloaders.append(dataloader)

    # Testing dataset
    holdout_rows = df.index[(df["holdout"]) & (df["manufacturer_id"] != -1)].tolist()
    df_subset = df.loc[holdout_rows]
    data = get_data_from_df(df=df_subset, data_dir=data_dir, targets=targets,
                            include_mappable_site_empty=include_mappable_site_empty)
    cache_dataset = get_dataset_from_data(data, cache_transforms, cache_type=cache_type,
                                          cache_dir=cache_dir)
    dataset = Dataset(data=cache_dataset, transform=valid_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloaders.append(dataloader)

    # Unknown dataset
    unknown_rows = df.index[df["manufacturer_id"] == -1].tolist()
    df_subset = df.loc[unknown_rows]
    data = get_data_from_df(df=df_subset, data_dir=data_dir, targets=targets,
                            include_mappable_site_empty=include_mappable_site_empty)
    cache_dataset = get_dataset_from_data(data, cache_transforms, cache_type=cache_type,
                                          cache_dir=cache_dir)
    dataset = Dataset(data=cache_dataset, transform=unknown_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloaders.append(dataloader)

    return dataloaders
