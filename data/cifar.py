import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class CIFAR10(Dataset):
    def __init__(self, root, transform=None, train=True):
        if root[0] == "~":
            # interprete `~` as the home directory.
            root = os.path.expanduser(root)
        self.root = root
        self.base_folder = "cifar-10-batches-py"
        self.transform = transform
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.train = train
        if self.train:
            data_list = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]
        else:
            data_list = ["test_batch"]
        data = []
        targets = []
        for file_name in data_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
            data.append(entry["data"])
            targets.extend(entry["labels"])
        # Convert data (List) to NHWC (np.ndarray) works with PIL Image.
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        self.data = data
        self.targets = np.asarray(targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)  # HWC ndarray->HWC Image.
        # HWC Image->CHW tensor.
        if self.transform is not None:
            img = self.transform(img)
        item = {"img": img, "target": target}

        return item

    def __len__(self):
        return len(self.data)
