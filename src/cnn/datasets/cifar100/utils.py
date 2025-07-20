import os
import pickle

from torchvision.datasets.utils import download_and_extract_archive


URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
BASE_FOLDER = 'cifar-100-python'
FILENAME = "cifar-100-python.tar.gz"
TGZ_MD5 = 'eb9058c3a382ffc7106e4002c42a8d85'


def load_cifar100_data(data_dir: str, url=URL, base_folder=BASE_FOLDER,
                       filename=FILENAME, md5=TGZ_MD5, train=True):
    download_and_extract_archive(url, download_root=data_dir, filename=filename, md5=md5)
    file_path = os.path.join(data_dir, base_folder, 'train' if train else 'test')
    with open(file_path, 'rb') as f:
        entry = pickle.load(f, encoding='latin1')
    images = entry['data']
    coarse_labels = entry['coarse_labels']
    fine_labels = entry['fine_labels']
    return images, coarse_labels, fine_labels
