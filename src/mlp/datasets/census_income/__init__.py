from typing import List, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader, random_split

from .utils import CensusIncomeDataset, CATEGORICAL_ATTRS_CLASSES, NUMERICAL_ATTRS, TARGETS_CLASSES


def get_census_income_dataloaders(df_train: pd.DataFrame, df_test: pd.DataFrame,
                                  categorical_attrs_classes: Optional[dict] = CATEGORICAL_ATTRS_CLASSES,
                                  numerical_attrs: Optional[List[str]] = NUMERICAL_ATTRS,
                                  targets_classes: Optional[dict] = TARGETS_CLASSES,
                                  targets: Optional[List[str]] = ["race", "native-country"],
                                  batch_size: int = 32,
                                  shuffle: bool = True,
                                  val_split: float = 0.1,
                                  num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:

    dataset = CensusIncomeDataset(df_train,
                                  categorical_attrs_classes=categorical_attrs_classes,
                                  numerical_attrs=numerical_attrs,
                                  targets_classes=targets_classes,
                                  targets=targets)
    num_val = int(len(dataset) * val_split)
    data_val, data_train = random_split(dataset, [num_val, len(dataset) - num_val])

    dataset_test = CensusIncomeDataset(df_test,
                                       categorical_attrs_classes=categorical_attrs_classes,
                                       numerical_attrs=numerical_attrs,
                                       targets_classes=targets_classes,
                                       targets=targets)

    dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader_train, dataloader_val, dataloader_test
