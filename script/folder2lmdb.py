import argparse
import json
import os
import pickle
import re

import lmdb
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm


def get_du_dir(dir):
    """Get the disk usage of ``dir`` recursively."""
    # Use ``st_blocks`` to get number of 512-byte blocks allocated for file rather than ``st_size``
    # which returns the apparent size.
    total_size = os.stat(dir).st_blocks * 512
    for entry in os.scandir(dir):
        if entry.is_file():
            total_size += entry.stat().st_blocks * 512
        elif entry.is_dir():
            total_size += get_du_dir(entry.path)

    return total_size


def single_folder2lmdb(
    img_dir: str, lmdb_dir: str, idx_2_class_path: str, write_freq: int = 5000
):
    """Convert the single image folder ``img_dir`` to the LMDB file ``lmdb_dir``.
    The images are arranged in this way: ::

        img_dir/0.png
        img_dir/1.png
        img_dir/[index].png
        img_dir/n.png
    And ``idx_2_class_path`` should be a json file with items (index, class).
    See :class:`LMDBDataset` for the LMDB file format. ``id`` is the index.

    Modified from https://github.com/Lyken17/Efficient-PyTorch.
    1. Use pickle to do serialization instead of pyarrow.
    2. Iterate the image folder directly instead of using the pytorch dataloader.
    3. Estimate the environment mapsize.
    """
    print("Loading the single image folder from {}".format(img_dir))
    print("Generate the LMDB file to {}".format(lmdb_dir))

    file_list = os.listdir(img_dir)
    for f in file_list:
        if not os.path.isfile(os.path.join(img_dir, f)):
            raise ValueError(
                "Each entry in dataset_dir: {} should be a file. "
                "Found {} is not.".format(img_dir, f)
            )
        if re.search("^\d+", f) is None:
            raise ValueError(
                "The filename f should start with a digital index pattern. "
                "Found {} is not.".format(f)
            )
    # Sort the filename by number in ascending order.
    file_list.sort(key=lambda f: int(re.findall("^\d+", f)[0]))
    file_path_list = [os.path.join(img_dir, f) for f in file_list]
    # The environment mapsize should be larger than it.
    size = get_du_dir(img_dir)

    db = lmdb.open(
        lmdb_dir,
        subdir=True,
        map_size=size * 100,
        readonly=False,
        meminit=False,
        map_async=True,
    )
    txn = db.begin(write=True)
    with open(idx_2_class_path) as f:
        idx_2_class = json.load(f)
    for idx, p in enumerate(tqdm(file_path_list)):
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        target = idx_2_class[str(idx)]
        txn.put(u"{}".format(idx).encode("ascii"), pickle.dumps((img, target)))
        if idx % write_freq == 0:
            txn.commit()
            txn = db.begin(write=True)

    # Finish iterating through dataset.
    txn.commit()
    keys = [u"{}".format(k).encode("ascii") for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__len__", pickle.dumps(len(keys)))

    print("Flushing database...")
    db.sync()
    db.close()


def image_folder2lmdb(img_dir: str, lmdb_dir: str, write_freq: int = 5000):
    """Convert the image folder ``img_dir`` to the LMDB file ``lmdb_dir``.
    The images are arranged like :class:`torchvision.datasets.ImageFolder`: ::

        img_dir/dog/xxx.png
        img_dir/dog/xxy.png
        img_dir/dog/[...]/xxz.png

        img_dir/cat/123.png
        img_dir/cat/nsdf3.png
        img_dir/cat/[...]/asd932_.png

    See :class:`LMDBDataset` for the LMDB file format. ``id`` is the filename without extension.

    Modified from https://github.com/Lyken17/Efficient-PyTorch.
    1. Use pickle to do serialization instead of pyarrow.
    2. Iterate the image folder directly instead of using the pytorch dataloader.
    3. Estimate the environment mapsize.
    """
    dataset = ImageFolder(img_dir)
    size = get_du_dir(img_dir)
    db = lmdb.open(
        lmdb_dir,
        subdir=True,
        map_size=size * 100,
        readonly=False,
        meminit=False,
        map_async=True,
    )
    txn = db.begin(write=True)
    filename_list = []
    for idx, (path, class_idx) in enumerate(tqdm(dataset.samples)):
        filename = os.path.splitext(os.path.basename(path))[0]
        filename_list.append(filename)
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
            txn.put(
                u"{}".format(filename).encode("ascii"), pickle.dumps((img, class_idx))
            )
        if idx % write_freq == 0:
            txn.commit()
            txn = db.begin(write=True)

    # Finish iterating through dataset.
    txn.commit()
    keys = [u"{}".format(f).encode("ascii") for f in filename_list]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", pickle.dumps(keys))
        txn.put(b"__len__", pickle.dumps(len(keys)))

    print("Flushing database...")
    db.sync()
    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir", help="Path to the original image folder.", type=str)
    parser.add_argument("lmdb_dir", help="Path to the output LMDB dir.", type=str)
    parser.add_argument("--folder-type", choices=["single", "image"], type=str)
    parser.add_argument(
        "--idx-2-class",
        default=None,
        help="Path to the json file with items (idx, class).",
    )
    args = parser.parse_args()
    if args.folder_type == "single":
        single_folder2lmdb(args.img_dir, args.lmdb_dir, args.idx_2_class)
    else:
        image_folder2lmdb(args.img_dir, args.lmdb_dir)
