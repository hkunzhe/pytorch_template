from torch.utils.data import DataLoader

from .cifar import CIFAR10
from .imagenet import ImageNet
from .prefetch import PrefetchLoader


def get_dataset(dataset_dir, transform, train=True):
    if "cifar-10" in dataset_dir:
        dataset = CIFAR10(dataset_dir, transform=transform, train=train)
    elif "imagenet" in dataset_dir:
        dataset = ImageNet(dataset_dir, transform=transform, train=train)
    else:
        raise NotImplementedError("Dataset in {} is not supported.".format(dataset_dir))

    return dataset


def get_loader(dataset, prefetch=False, loader_config=None, **kwargs):
    if loader_config is None:
        loader = DataLoader(dataset, **kwargs)
    else:
        loader = DataLoader(dataset, **loader_config, **kwargs)
    if prefetch:
        loader = PrefetchLoader(loader)

    return loader
