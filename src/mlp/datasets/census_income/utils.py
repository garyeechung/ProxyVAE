from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

CATEGORICAL_ATTRS_CLASSES = {
    "workclass": ["Federal-gov", "Local-gov", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"],
    "education": ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Some-college", "Assoc-voc", "Assoc-acdm", "Bachelors", "Masters", "Prof-school", "Doctorate"],
    "marital-status": ["Never-married", "Separated", "Married-spouse-absent", "Divorced", "Married-civ-spouse", "Married-AF-spouse", "Widowed"],
    "occupation": ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving"],
    "relationship": ["Own-child", "Not-in-family", "Other-relative", "Unmarried", "Wife", "Husband"],
    "sex": ["Female", "Male"],
    "income": ["<=50K", ">50K"],
}

NUMERICAL_ATTRS = [
    "age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"
]

TARGETS_CLASSES = {
    "race": ["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "White", "Other"],
    "native-country": ["Cambodia", "Canada", "China", "Columbia", "Cuba", "Dominican-Republic", "Ecuador", "El-Salvador", "England", "France", "Germany", "Greece", "Guatemala", "Haiti", "Holand-Netherlands", "Honduras", "Hong", "Hungary", "India", "Iran", "Ireland", "Italy", "Jamaica", "Japan", "Laos", "Mexico", "Nicaragua", "Outlying-US(Guam-USVI-etc)", "Peru", "Philippines", "Poland", "Portugal", "Puerto-Rico", "Scotland", "South", "Taiwan", "Thailand", "Trinadad&Tobago", "United-States", "Vietnam", "Yugoslavia", ],
}


class CensusIncomeDataset(Dataset):
    def __init__(self, df: pd.DataFrame,
                 categorical_attrs_classes: Dict[str, List[str]] = CATEGORICAL_ATTRS_CLASSES,
                 numerical_attrs: List[str] = NUMERICAL_ATTRS,
                 targets_classes: Dict[str, List[str]] = TARGETS_CLASSES,
                 targets: List[str] = ["race", "native-country"],
                 one_hot: bool = True):
        self.df = df.reset_index(drop=True)

        self.categorical_attrs_classes = categorical_attrs_classes
        self.categorical_attrs = list(categorical_attrs_classes.keys())
        self.numerical_attrs = numerical_attrs
        self.targets_classes = targets_classes
        self.targets = targets
        for attr in self.categorical_attrs + self.numerical_attrs + self.targets:
            assert attr in df.columns, f"Attribute {attr} not found in DataFrame columns"

        data = []
        for _, row in self.df.iterrows():
            item = []
            x = []
            for attr in self.categorical_attrs:
                x_cat = row[attr]
                cat_list = self.categorical_attrs_classes[attr]
                x_enc = self.get_encoding(x_cat, cat_list, one_hot=one_hot)
                x.append(x_enc)
            for attr in self.numerical_attrs:
                x_num = row[attr]
                x.append(np.array([x_num], dtype=np.float32))
            x = np.concatenate(x).astype(np.float32)
            x = torch.from_numpy(x)
            item.append(x)

            for target in self.targets:
                y_cat = row[target]
                target_list = self.targets_classes[target]
                y_enc = self.get_encoding(y_cat, target_list, one_hot=one_hot)
                item.append(y_enc)
            data.append(item)

        self.data = data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_encoding(self, x_cat, cat_list, one_hot=True):
        num_classes = len(cat_list)
        cat_list = np.array(cat_list)
        cat_idx = np.where(cat_list == x_cat)[0][0]
        if one_hot:
            x_one_hot = np.eye(num_classes)[cat_idx]
            return x_one_hot
        else:
            return cat_idx
