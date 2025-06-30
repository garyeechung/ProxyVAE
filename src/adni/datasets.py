from typing import List, Optional

from monai.transforms import Compose, CropForeground, Lambdad
from monai.transforms import Resized, ResizeWithPadOrCrop, MapTransform
from monai.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import pandas as pd


class LoadAndPreprocessSlice(MapTransform):
    def __init__(self, keys, slice_range_from_center: float = 0.0, margin: int = 10):
        super().__init__(keys=keys)
        assert 0.0 <= slice_range_from_center <= 1.0, "ratio must be between 0.0 and 1.0"
        self.slice_range_from_center = slice_range_from_center
        assert margin >= 0, "margin must be non-negative integer"
        self.margin = margin

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            path = d[key]
            volume = nib.load(path).get_fdata()
            volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=0.0)
            volume = np.clip(volume, 0.0, 1.0)
            volume = volume.transpose(2, 1, 0)  # from HWD to DWH
            volume = self._crop_foreground(volume)

            nb_slices = volume.shape[0]
            center_slice = nb_slices // 2
            slice_range = int(self.slice_range_from_center * nb_slices / 2)
            start_slice = max(0, center_slice - slice_range)
            end_slice = min(nb_slices, center_slice + slice_range)
            if start_slice >= end_slice:
                slice_index = center_slice
            else:
                slice_index = np.random.randint(start_slice, end_slice)
            slice_ = volume[slice_index:slice_index + 1, :, :]
            slice_ = self._crop_foreground(slice_)
            d[key] = slice_

        return d

    def _crop_foreground(self, image: np.ndarray) -> np.ndarray:
        crop_foreground = CropForeground(select_fn=lambda x: x > 0., margin=self.margin)
        return crop_foreground(image)


class PadToSquare(MapTransform):
    def __init__(self, keys=None):
        super().__init__(keys=keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            h, w = img.shape[-2:]
            side = max(h, w)
            padder = ResizeWithPadOrCrop(spatial_size=(side, side))
            d[key] = padder(img)
        return d


class OneHotEncoded(MapTransform):
    def __init__(self, keys: List[str], num_classes: List[int]):
        super().__init__(keys=keys)
        self.num_classes = num_classes
        assert len(keys) == len(num_classes), "keys and num_classes must have the same length"

    def __call__(self, data):
        d = dict(data)
        for key, nc in zip(self.keys, self.num_classes):
            lab = d[key]
            d[key] = np.eye(nc)[int(lab)]
        return d


def split_train_val(list_):
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


def get_data_from_df(df, targets: List[str], include_mappable_site_empty=False):

    if not include_mappable_site_empty:
        df = df[~df["site_empty"]]

    data = []
    for _, row in df.iterrows():
        item = {"image": row["full_path"]}
        for target in targets:
            item[target] = row[target]
        data.append(item)

    return data


def get_adni_dataloaders(df, targets: List[str],
                         stratified_by: Optional[List[str]] = ["model_type_id"],
                         bootstrap=True, batch_size=32, num_workers=4,
                         include_mappable_site_empty=False, seed=42):
    np.random.seed(seed)
    AVAILABLE_TARGETS = ["manufacturer_id", "model_type_id", "site"]
    for target in targets:
        assert target in AVAILABLE_TARGETS, f"target must be one of {AVAILABLE_TARGETS}, got {target}"

    num_classes = [df[target].max() + 1 for target in targets]

    transforms_train = Compose([
        LoadAndPreprocessSlice(keys=["image"], slice_range_from_center=0.1, margin=10),
        PadToSquare(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128), mode="bilinear", align_corners=False),
        Lambdad(keys=["image"], func=lambda x: np.clip(x, 0.0, 1.0)),
        OneHotEncoded(keys=targets, num_classes=num_classes)
    ])

    transforms_valid = Compose([
        LoadAndPreprocessSlice(keys=["image"], slice_range_from_center=0.0, margin=10),
        PadToSquare(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128), mode="bilinear", align_corners=False),
        Lambdad(keys=["image"], func=lambda x: np.clip(x, 0.0, 1.0)),
        OneHotEncoded(keys=targets, num_classes=num_classes)
    ])

    transforms_unknown = Compose([
        LoadAndPreprocessSlice(keys=["image"], slice_range_from_center=0.0, margin=10),
        PadToSquare(keys=["image"]),
        Resized(keys=["image"], spatial_size=(128, 128), mode="bilinear", align_corners=False),
        Lambdad(keys=["image"], func=lambda x: np.clip(x, 0.0, 1.0))
    ])

    unknown_rows = df.index[df["manufacturer_id"] == -1].tolist()
    holdout_rows = df.index[(df["holdout"]) & (df["manufacturer_id"] != -1)].tolist()

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
        if bootstrap:
            max_size = df_train_val_split["size"].max()
            df_train_val_split = df_train_val_split.assign(bootstrap=lambda x: x["size"].apply(lambda size: max_size * 5 // size))
        else:
            df_train_val_split = df_train_val_split.assign(bootstrap=1)
        train_rows = df_train_val_split.apply(lambda x: x["train_ids"] * x["bootstrap"], axis=1).sum()
        valid_rows = df_train_val_split.apply(lambda x: x["valid_ids"] * x["bootstrap"], axis=1).sum()

    else:
        rest_ids = df.index[~df["holdout"]].tolist()
        rest_ids = np.random.permutation(rest_ids)
        train_size = int(len(rest_ids) * 0.9)
        train_rows = rest_ids[:train_size]
        valid_rows = rest_ids[train_size:]

    dataloaders = []

    # Training Dataloader
    df_subset = df.loc[train_rows]
    data = get_data_from_df(df_subset, targets, include_mappable_site_empty)
    dataset = Dataset(data=data, transform=transforms_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    dataloaders.append(dataloader)

    # Validation Dataloader
    df_subset = df.loc[valid_rows]
    data = get_data_from_df(df_subset, targets, include_mappable_site_empty)
    dataset = Dataset(data=data, transform=transforms_valid)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    dataloaders.append(dataloader)

    # Testing Dataloader
    df_subset = df.loc[holdout_rows]
    data = get_data_from_df(df_subset, targets, include_mappable_site_empty)
    dataset = Dataset(data=data, transform=transforms_valid)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    dataloaders.append(dataloader)

    # Unknown Dataloader
    df_subset = df.loc[unknown_rows]
    data = [{"image": row["full_path"]} for _, row in df_subset.iterrows()]
    dataset = Dataset(data=data, transform=transforms_unknown)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    dataloaders.append(dataloader)

    return dataloaders
