from torch.utils.data import DataLoader, random_split
from .utils import MNISTCoarseFineDataset, FINE_COARSE_MAP, COARSE_FINE_MAP


def get_mnist_dataloaders(root: str, batch_size=50, num_workers=4, one_hot=True):
    mnist = MNISTCoarseFineDataset(root=root, train=True, download=True, one_hot=one_hot)
    mnist_test = MNISTCoarseFineDataset(root=root, train=False, download=True, one_hot=one_hot)
    mnist_train, mnist_val = random_split(mnist, [50000, 10000])
    mnist_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    mnist_val = DataLoader(mnist_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    mnist_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return mnist_train, mnist_val, mnist_test


__all__ = ['get_mnist_dataloaders', 'FINE_COARSE_MAP', 'COARSE_FINE_MAP']
