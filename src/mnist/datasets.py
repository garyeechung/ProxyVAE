import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.datasets as datasets


def get_mnist_dataloaders(root: str, batch_size=50, num_workers=4, one_hot=True):
    """
    Load MNIST dataset and return train and test dataloaders.
    """
    transform_x = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
        lambda x: torch.flatten(x)
    ])

    transform_y = [lambda x:torch.LongTensor([x])]
    if one_hot:
        transform_y.append(lambda x: F.one_hot(x, 10))
    transform_y.append(lambda x: torch.flatten(x))
    transform_y = T.Compose(transform_y)

    mnist = datasets.MNIST(root, train=True, download=True,
                           transform=transform_x,
                           target_transform=transform_y)
    mnist_test = datasets.MNIST(root, train=False, download=True,
                                transform=transform_x,
                                target_transform=transform_y)
    mnist_train, mnist_val = data.random_split(mnist, [50000, 10000])

    mnist_train = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_val = data.DataLoader(mnist_val, batch_size=batch_size, shuffle=False)
    mnist_test = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return mnist_train, mnist_val, mnist_test


def convert_flattened_to_image(x):
    """
    Convert a flattened MNIST image back to its original shape.
    """
    return x.view(-1, 28, 28)


MERGE_GROUP = [
    [0, 6],
    [1],
    [4, 7, 9],
    [2, 3, 5, 8]
]


def get_merged_labels(labels, merge_group=MERGE_GROUP, one_hot=True):

    # Build a mapping from label to group index
    label_to_group = {label: i for i, group in enumerate(merge_group) for label in group}

    # Map each label using the dictionary
    merged_labels = torch.tensor([label_to_group[label.item()] for label in labels], device=labels.device)

    if one_hot:
        # Convert to one-hot encoding
        merged_labels = F.one_hot(merged_labels, num_classes=len(merge_group))
        merged_labels = merged_labels.view(-1, len(merge_group))

    return merged_labels
