import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASETS = {
    "mnist": datasets.MNIST,
    "cifar": datasets.CIFAR10
}

def get_dataset(dataset_name, train=False):
    if dataset_name not in DATASETS:
        raise NameError('Dataset is not presented. Valid dataset names are {}'.format(list(DATASETS.keys())))
    path_to_save = os.getcwd() + "/data"
    return DataLoader(DATASETS[dataset_name](path_to_save, train=train, download=True,
                                                    transform=transforms.Compose([transforms.ToTensor()])),
                             batch_size=1, shuffle=True)
