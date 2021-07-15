import warnings

import kornia as K
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.functional import _interpolation_modes_from_int


def get_mean_std(transform):
    """Get mean and standard deviation from ``transform``."""
    if not isinstance(
        transform, (transforms.Compose, TorchTransforms, KorniaTransforms)
    ):
        raise TypeError(
            "Transform should be transforms.Compose, TorchTransforms or "
            "KorniaTransforms. Got {}.".format(type(transform))
        )
    normalize = False
    for t in transform.transforms:
        if isinstance(t, (transforms.Normalize, K.enhance.Normalize)):
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
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self.inplace = inplace

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Tensor should be torch.Tensor. Got {}".format(type(tensor))
            )
        if len(tensor.shape) != 4:
            raise ValueError("Tensor should be with (B, C, H, W) shape.")
        if not self.inplace:
            tensor = tensor.clone()
        mean = self.mean.clone().to(tensor.device).view(1, tensor.shape[1], 1, 1)
        std = self.std.clone().to(tensor.device).view(1, tensor.shape[1], 1, 1)
        # TODO: numerical stability.
        tensor = tensor.mul(std).add(mean)

        return tensor


class TorchTransforms(nn.Module):
    """A ``torchvision.transforms.Compose`` wrapper supports a transformation
    configuration dict.

    Args:
        transform_config (dict): The transformation configuration dict.
    """

    def __init__(self, transform_config):
        super(TorchTransforms, self).__init__()
        self.transform_config = transform_config
        self.transform = self._get_transform(transform_config)  # transforms.Compose
        # This is consistent with ``transforms.Compose`` has a ``transforms``
        # attribute (a list of transforms).
        self.transforms = self.transform.transforms

    def _get_transform(self, transform_config):
        """Convert a transformation configuration dict to ``transforms.Compose``."""
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
            # Backward compatibility with integer value.
            if "interpolation" in kwargs:
                kwargs["interpolation"] = _interpolation_modes_from_int(
                    kwargs["interpolation"]
                )
            return transforms.RandomResizedCrop(**kwargs)
        elif name == "resize":
            # Backward compatibility with integer value.
            if "interpolation" in kwargs:
                kwargs["interpolation"] = _interpolation_modes_from_int(
                    kwargs["interpolation"]
                )
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
    """A ``Kornia`` wrapper supports a transformation configuration dict.

    Args:
        transform_config (dict): The transformation configuration dict.
    """

    def __init__(self, transform_config):
        super(KorniaTransforms, self).__init__()
        self.transform_config = transform_config
        self.transform = self._get_transform(transform_config)  # nn.Sequential
        # This is consistent with ``transforms.Compose`` has ``transforms``
        # attribute (a list of transforms).
        self.transforms = list(self.transform)

    def _get_transform(self, transform_config):
        """Convert a transformation configuration dict to ``nn.Sequential``."""
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
            if isinstance(kwargs["mean"], (list, tuple)):
                kwargs["mean"] = torch.tensor(kwargs["mean"])
            if isinstance(kwargs["std"], (list, tuple)):
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


def concat_transform(*transform):
    """Concat a squence of transform.

    Args:
       transform (sequence): A list or tuple of TorchTransforms, KorniaTransforms or
           transforms.Compose.
    """
    t_type = [type(t) for t in transform]
    same_type = all(t == t_type[0] for t in t_type)
    if not same_type:
        raise TypeError(
            "All elements in transform should be with the same type. "
            "Got {}".format(t_type)
        )

    concated_transform = []
    for t in transform:
        concated_transform.extend(t.transforms)
    if isinstance(transform[0], (TorchTransforms, transforms.Compose)):
        concated_transform = transforms.Compose(concated_transform)
    elif isinstance(transform[0], KorniaTransforms):
        concated_transform = nn.Sequential(*concated_transform)
    else:
        raise TypeError(
            "Elements in transform should be TorchTransforms, "
            "transforms.Compose or KorniaTransforms. Got {}".format(type(transform[0]))
        )

    return concated_transform
