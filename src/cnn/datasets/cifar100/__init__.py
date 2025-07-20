import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .utils import load_cifar100_data


class CIFAR100Dataset(Dataset):
    def __init__(self, data_dir, train=True, one_hot=True):
        self.train = train
        self.one_hot = one_hot
        self.images, self.coarse_labels, self.fine_labels = load_cifar100_data(data_dir, train=train)
        self.images = self.images.reshape((-1, 3, 32, 32)).astype(np.float32) / 255.0
        self.images = torch.from_numpy(self.images)
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.images = (self.images - imagenet_mean) / imagenet_std

        if self.one_hot:
            self.coarse_labels = np.eye(20)[self.coarse_labels].astype(np.float32)
            self.fine_labels = np.eye(100)[self.fine_labels].astype(np.float32)
        self.coarse_labels = torch.from_numpy(self.coarse_labels)
        self.fine_labels = torch.from_numpy(self.fine_labels)

        # Sort the dataset by fine labels
        sorted_indices = np.argsort(self.fine_labels.numpy().argmax(axis=1))
        self.images = self.images[sorted_indices]
        self.coarse_labels = self.coarse_labels[sorted_indices]
        self.fine_labels = self.fine_labels[sorted_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        coarse_label = self.coarse_labels[idx]
        fine_label = self.fine_labels[idx]
        data_dict = {
            'image': image,
            'coarse_label': coarse_label,
            'fine_label': fine_label
        }
        return data_dict


def get_cifar100_dataloaders(data_dir: str, batch_size: int = 32, val_ratio: float = 0.2):
    dataset = CIFAR100Dataset(data_dir, train=True)
    dataset_tst = CIFAR100Dataset(data_dir, train=False)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    fine_label_indices = dataset.fine_labels.numpy().argmax(axis=1)
    train_idx, val_idx = next(splitter.split(np.zeros(len(dataset)), fine_label_indices))

    dataset_trn = Subset(dataset, train_idx)
    dataset_val = Subset(dataset, val_idx)

    dataloader_trn = DataLoader(dataset_trn, batch_size=batch_size, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)
    dataloader_tst = DataLoader(dataset_tst, batch_size=batch_size, shuffle=False, num_workers=4)

    return dataloader_trn, dataloader_val, dataloader_tst
