import warnings

import torch
import torchvision.transforms as transforms

from .collate import pil_collate


def prefetch_transform(transform):
    """Remove `ToTensor` and `Normalize` in `transform`."""
    transform_list = []
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
    for t in transform.transforms:
        if not ("ToTensor" in str(type(t)) or "Normalize" in str(type(t))):
            transform_list.append(t)
    remaining_transform = transforms.Compose(transform_list)

    return remaining_transform, mean, std


class PrefetchLoader:
    """A DataLoader wrapper for prefetching data to GPUs along with
    ``ToTensor`` and ``Normalize`` transformations (torchvision).

    FIXME: Turn on prefetch mode will increase GPU memory every epoch until filled
    and restart again, especially in training ImageNet models and CIFAR-10 models
    with large batch size (2048).

    Reference:
    [1] https://github.com/NVIDIA/apex/tree/master/examples/imagenet.
    [2] https://github.com/open-mmlab/OpenSelfSup.
    """

    def __init__(self, loader):
        self.loader = loader
        transform = self.dataset.transform  # TorchTransforms or transforms.Compose
        self.mean, self.std = None, None
        if transform is not None:
            remaining_transform, self.mean, self.std = prefetch_transform(transform)
            self.dataset.transform = remaining_transform  # transforms.Compose
        # The item in ``self.loader`` may be class ``PIL.Image.Image``, since
        # the ``ToTensor`` transformation are preformed along with prefetching.
        self.loader.collate_fn = pil_collate
        self.normalize = (self.mean is not None) and (self.std is not None)
        if self.normalize:
            self.mean = torch.as_tensor(self.mean)
            self.std = torch.as_tensor(self.std)

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True
        if self.normalize:
            self.mean = self.mean.cuda().view(1, 3, 1, 1)
            self.std = self.std.cuda().view(1, 3, 1, 1)

        for next_item in self.loader:
            with torch.cuda.stream(stream):
                if "img" in next_item:
                    img = next_item["img"].cuda(non_blocking=True)
                    img = img.float().div(255)
                    if self.normalize:
                        img = img.sub(self.mean).div(self.std)
                    next_item["img"] = img
                else:
                    raise KeyError("The item in self.loader must contain key img.")

            if not first:
                yield item
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            item = next_item

        yield item

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset
