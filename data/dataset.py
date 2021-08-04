import os
import pickle

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class LMDBDataset(Dataset):
    """Pytorch dataset for the LMDB file, which has the following format:

    ===========  =========================
    key          value
    ===========  =========================
    id1          (PIL Image 1, int label1)
    id2          (PIL Image 2, int label2)
    [...]        [...]
    idn          (PIL Image n, int labeln)
    ``__keys__`` [id1, id2, [...], idn]
    ``__len__``  n
    ===========  =========================

    ``id`` can be the index, filename, etc. Since data are loaded dynamically, ``__keys__`` and
    ``__len__`` are necessary for :method:`__get_item__`.

    .. note::
        To avoid the `TypeError: can't pickle Environment objects` when using DDP:
        1. | Open an LMDB environment in :method:`__init__` to load meta informations (the length
           | and keys), just be sure to close it within :method:`__init__`.
        2. | Initilize the LMDB environment at the first data iteration.

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
            # Deserialize keys (a long byteflow) in initialization since is time-consuming.
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
        img, target = pickle.loads(byteflow)  # PIL Image, int
        assert isinstance(img, Image.Image)
        if self.transform is not None:
            img = self.transform(img)
        item = {"img": img, "target": target}

        return item

    def __len__(self):
        return self.length
