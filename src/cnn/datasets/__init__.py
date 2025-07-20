from .adni import get_adni_dataloaders
from .cifar100 import get_cifar100_dataloaders


if __name__ == "__main__":
    get_adni_dataloaders.__doc__ = "This module provides functions to load ADNI datasets."
    get_cifar100_dataloaders.__doc__ = "This module provides functions to load CIFAR-100 datasets."
