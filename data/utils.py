import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_transform(transform_config):
    transform = []
    if "random_resize_crop" in transform_config:
        transform.append(
            transforms.RandomResizedCrop(**transform_config["random_resize_crop"])
        )
    if "random_horizontal_flip" in transform_config:
        transform.append(transforms.RandomHorizontalFlip(**transform_config["hflip"]))
    transform.append(transforms.ToTensor())
    if "normalize" in transform_config:
        transform.append(transforms.Normalize(**transform_config["normalize"]))

    return transforms.Compose(transform)


def get_loader(dataset, loader_config, **kwargs):
    return DataLoader(dataset, **loader_config, **kwargs)
