import argparse
import json
import os
import pickle
import re

import lmdb
from PIL import Image


def single_folder2lmdb(
    img_dir: str, lmdb_dir: str, idx_2_class_path: str, write_frequency: int = 5000
):
    """Convert the single image folder ``img_dir`` to the LMDB file ``lmdb_dir``.
    The images are arranged in this way: ::

        img_dir/0.png
        img_dir/1.png
        img_dir/[index].png
        img_dir/n.png
    And ``idx_2_class_path`` should be a json file with items (index, class).
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
    # Estimate the environment mapsize. Use ``st_blacks`` not ``st_size`` to be
    # consistent with ``du`` which shows the size in 512-byte blocks.
    size = sum(os.stat(f).st_blocks * 512 for f in os.scandir(img_dir))

    db = lmdb.open(
        lmdb_dir,
        subdir=True,
        map_size=size * 2,
        readonly=False,
        meminit=False,
        map_async=True,
    )
    txn = db.begin(write=True)
    with open(idx_2_class_path) as f:
        idx_2_class = json.load(f)
    for idx, p in enumerate(file_path_list):
        with open(p, "rb") as f:
            img = Image.open(f).convert("RGB")
        target = idx_2_class[str(idx)]
        txn.put(u"{}".format(idx).encode("ascii"), pickle.dumps((img, target)))
        if idx % write_frequency == 0:
            print("[{}/{}]".format(idx, len(file_path_list)))
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to the original image folder.")
    parser.add_argument("--out", help="Path to the output LMDB file.")
    parser.add_argument(
        "--idx-2-class", help="Path to the file with items (idx, class)."
    )
    args = parser.parse_args()
    single_folder2lmdb(args.dataset, args.out, args.idx_2_class)
