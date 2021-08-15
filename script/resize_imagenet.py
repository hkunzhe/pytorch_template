import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

num_workers = 16
dataset_dir = "/home/hkz/dataset/ILSVRC2012/val"
saved_dir = "/home/hkz/dataset/ILSVRC2012_preprocessed/val"

pre_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
dataset = ImageFolder(dataset_dir, transform=pre_transform)
data_loader = DataLoader(dataset, num_workers=num_workers, collate_fn=lambda x: x)
for idx, data in enumerate(data_loader):
    img, class_idx = data[0]  # PIL Image, int
    path = dataset.samples[idx][0]
    basename = os.path.basename(path)
    filename, _ = os.path.splitext(os.path.basename(basename))
    class_dir = path.split("/")[-2]
    pre_img_dir = os.path.join(saved_dir, class_dir)
    if not os.path.exists(pre_img_dir):
        os.mkdir(pre_img_dir)
    pre_img_path = os.path.join(pre_img_dir, filename + ".png")
    print("[{}/{}] Resizing {} to {}".format(idx, len(dataset), path, pre_img_path))
    img.save(pre_img_path)
