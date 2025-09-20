import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import MNIST


TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,)),
    lambda x: torch.flatten(x)
])

MNIST_MERGE_GROUP = COARSE_FINE_MAP = {
    0: [0, 6],  # coarse class 0: fine classes 0 and 6
    1: [1],     # coarse class 1: fine class 1
    2: [4, 7, 9],  # coarse class 2: fine classes 4, 7, and 9
    3: [2, 3, 5, 8]  # coarse class 3: fine classes 2, 3, 5, and 8
}

FINE_COARSE_MAP = {fine: coarse for coarse, fines in COARSE_FINE_MAP.items() for fine in fines}


class MNISTCoarseFineDataset(Dataset):
    def __init__(self, root: str, train: bool = True, download: bool = True,
                 coarse_fine_map=COARSE_FINE_MAP, one_hot: bool = False):
        super(MNISTCoarseFineDataset, self).__init__()
        transform = T.Compose([
            T.ToTensor(),
            # T.Normalize((0.1307,), (0.3081,)),
            lambda x: torch.flatten(x)
        ])
        # target_transform = [lambda x: torch.LongTensor([x])]
        self.mnist = MNIST(root=root, train=train, download=download, transform=transform)
        self.fine_coarse_map = {fine: coarse for coarse, fines in coarse_fine_map.items() for fine in fines}
        self.one_hot = one_hot
        self.num_coarse_classes = len(coarse_fine_map)
        self.num_fine_classes = 10

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        x, yf = self.mnist[idx]
        yc = self.fine_coarse_map[yf]
        if self.one_hot:
            yf = np.eye(self.num_fine_classes)[yf]
            yc = np.eye(self.num_coarse_classes)[yc]
            yf = torch.tensor(yf, dtype=torch.float32)
            yc = torch.tensor(yc, dtype=torch.float32)
        else:
            yf = torch.tensor(yf, dtype=torch.long)
            yc = torch.tensor(yc, dtype=torch.long)
        return x, yc, yf


if __name__ == "__main__":
    dataset = MNISTCoarseFineDataset(root='./data', train=True, download=True, one_hot=True)
    print(len(dataset))
    x, yf, yc = dataset[0]
    print(x.shape, yf.shape, yc.shape)
    print(yf, yc)
    print(yf.argmax().item(), yc.argmax().item())


def convert_flattened_to_image(x):
    """
    Convert a flattened MNIST image back to its original shape.
    """
    return x.view(-1, 28, 28)
