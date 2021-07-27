import os
import pickle

import lmdb
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    """The LMDB file dataset for Pytorch Dataset, which has the format:

    ===========  =======================
    key          value
    ===========  =======================
    id1          (PIL Image 1, label1)
    id2          (PIL Image 2, label2)
    \.\.\.       \.\.\.
    idn          (PIL Image n, labeln)
    ``__keys__`` [id1, id2, \.\.\., idn]
    ``__len__``  n
    ===========  =======================

    To avoid the `TypeError: can't pickle Environment objects` when using DDP:
    1. Open an LMDB environment in ``__init__`` to load meta informations (the length
    and keys of the dataset) just be sure to close it within ``__init__``.
    2. Initilize the LMDB environment at the first data iteration.

    Reference:
    1. https://github.com/Lyken17/Efficient-PyTorch.
    2. https://github.com/thecml/Efficient-PyTorch.
    3. https://github.com/pytorch/vision/issues/689.
    """

    def __init__(self, root, transform=None, train=True):
        if root[0] == "~":
            # interprete `~` as the home directory.
            root = os.path.expanduser(root)
        self.root = root
        self.train = train
        folder = "train" if train else "val"
        self.data_dir = os.path.join(self.root, folder)
        self.transform = transform
        env = lmdb.open(
            self.data_dir,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))
        env.close()

    def __getitem__(self, index):
        if not hasattr(self, "env"):
            self.env = lmdb.open(
                self.data_dir,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)
        img, target = unpacked[0], unpacked[1]  # PIL Image, int
        if self.transform is not None:
            img = self.transform(img)
        item = {"img": img, "target": target}

        return item

    def __len__(self):
        return self.length
