import warnings

import kornia as K
import torch
import torch.nn as nn
import torchvision.transforms as transforms


def get_mean_std(transform):
    """Get mean and standard deviation from ``transform``."""
    normalize = False
    for t in transform.transforms:
        if "Normalize" in str(type(t)):
            normalize = True
            mean, std = t.mean, t.std
    if not normalize:
        warnings.warn(
            "No Normalize in transform: {}. "
            "Return None for mean and standard deviation".format(transform)
        )
        mean, std = None, None

    return mean, std


class DeNormalize(object):
    """Torchvision-style reverse normalization for ``torch.Tensor`` with
    original mean and standard deviation value similar to
    ``kornia.enhance.Denormalize``.

    Args:
        mean (tuple or list): Mean for each channel.
        std (tuple or list): Standard deviation for each channel.
        inplace(bool): Bool to make this operation inplace. Default is False.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Tensor should be torch.Tensor. Got {}".format(type(tensor))
            )
        if not self.inplace:
            tensor = tensor.clone()
        # TODO: numerical stability.
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)

        return tensor


class TorchTransforms(nn.Module):
    def __init__(self, transform_config):
        super(TorchTransforms, self).__init__()
        self.transform_config = transform_config
        self.transform = self._get_transform(transform_config)
        # A list of transforms. which is consistent with ``transforms.Compose``.
        self.transforms = self.transform.transforms

    def _get_transform(self, transform_config):
        transform = []
        if transform_config is not None:
            for (k, v) in transform_config.items():
                if v is not None:
                    transform.append(self._query_transform(k, v))
        transform = transforms.Compose(transform)

        return transform

    def _query_transform(self, name, kwargs):
        if name == "random_crop":
            return transforms.RandomCrop(**kwargs)
        elif name == "random_resize_crop":
            return transforms.RandomResizedCrop(**kwargs)
        elif name == "resize":
            return transforms.Resize(**kwargs)
        elif name == "center_crop":
            return transforms.CenterCrop(**kwargs)
        elif name == "random_horizontal_flip":
            return transforms.RandomHorizontalFlip(**kwargs)
        elif name == "to_tensor":
            if kwargs:
                return transforms.ToTensor()
        elif name == "normalize":
            return transforms.Normalize(**kwargs)
        else:
            raise ValueError("Transformation {} is not supported!".format(name))

    def forward(self, x):
        x = self.transform(x)

        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"

        return format_string


class KorniaTransforms(nn.Module):
    def __init__(self, transform_config):
        super(KorniaTransforms, self).__init__()
        self.transform_config = transform_config
        self.transform = self._get_transform(transform_config)
        # A list of transforms. which is consistent with ``transforms.Compose``.
        self.transforms = list(self.transform)

    def _get_transform(self, transform_config):
        transform = []
        if transform_config is not None:
            for (k, v) in transform_config.items():
                if v is not None:
                    transform.append(self._query_transform(k, v))
        transform = nn.Sequential(*transform)

        return transform

    def _query_transform(self, name, kwargs):
        if name == "random_crop":
            return K.augmentation.RandomCrop(**kwargs)
        elif name == "random_resize_crop":
            return K.augmentation.RandomResizedCrop(**kwargs)
        elif name == "resize":
            return K.geometry.Resize(**kwargs)
        elif name == "center_crop":
            return K.augmentation.CenterCrop(**kwargs)
        elif name == "random_horizontal_flip":
            return K.augmentation.RandomHorizontalFlip(**kwargs)
        elif name == "normalize":
            if isinstance(kwargs["mean"], list) or isinstance(kwargs["mean"], tuple):
                kwargs["mean"] = torch.tensor(kwargs["mean"])
            if isinstance(kwargs["std"], list) or isinstance(kwargs["std"], tuple):
                kwargs["std"] = torch.tensor(kwargs["std"])
            return K.enhance.Normalize(**kwargs)
        else:
            raise ValueError("Kornia transformation {} is not supported!".format(name))

    # @torch.no_grad()  # disable gradients for effiency.
    def forward(self, x):
        x = self.transform(x)

        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"

        return format_string
