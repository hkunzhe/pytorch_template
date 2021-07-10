"""Modified from torch/utils/data/_utils/collate.py:
[1]. Add a case to handle class ``PIL.Image.Image``.
"""

import collections
import re

import numpy as np
import torch
from PIL import Image
from torch._six import string_classes

# fmt: off
np_str_obj_array_pattern = re.compile(r'[SaUO]')

pil_collate_err_msg_format = (
    "pil_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def pil_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(pil_collate_err_msg_format.format(elem.dtype))

            return pil_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: pil_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(pil_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [pil_collate(samples) for samples in transposed]
    elif isinstance(elem, Image.Image):
        # Handle class ``PIL.Image.Image``.
        converted_batch = []
        for img in batch:
            img = np.array(img, dtype=np.uint8)
            if img.ndim == 2:
                img = img[:, :, None]
            converted_batch.append(img.transpose((2, 0, 1)))
        return pil_collate(converted_batch)
