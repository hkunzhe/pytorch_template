import json
import os

from tqdm import tqdm

from .data import cifar

ROOT = "/home/hkz/dataset/cifar-10_img"
TRAIN = True
FOLDER = "train" if TRAIN else "val"
DATA_DIR = os.path.join(ROOT, FOLDER)
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
print("Convert the cifar-10 dataset to a single folder {}".format(DATA_DIR))

cifar10_data = cifar.CIFAR10("~/dataset/cifar-10", train=TRAIN)
idx_2_class = {}
for i in tqdm(range(len(cifar10_data))):
    idx_2_class[str(i)] = int(cifar10_data[i]["target"])
    saved_path = os.path.join(DATA_DIR, str(i) + ".png")
    cifar10_data[i]["img"].save(saved_path)

idx_2_class_path = os.path.join(ROOT, "idx_2_class_{}.json".format(FOLDER))
with open(idx_2_class_path, "w") as f:
    json.dump(idx_2_class, f)
print("Save items (index class) to {}".format(idx_2_class_path))
