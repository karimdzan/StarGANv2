import os
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
import csv
import numpy as np
import random
from src.datasets import BaseDataset


class CelebADataset(BaseDataset):
    def __init__(self, root_dir, transform=None, limit=None):
        """
        Args:
          root_dir (string): Directory with all the images
          transform (callable, optional): transform to be applied to each image sample
        """
        # Read names of images in the root directory

        # Path to folder with the dataset
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        dataset_folder = f"{root_dir}/img_align_celeba/"
        self.dataset_folder = os.path.abspath(dataset_folder)
        image_names = os.listdir(self.dataset_folder)

        self.transform = transform
        image_names = natsorted(image_names)

        self.filenames = []
        self.annotations = []
        with open(f"{root_dir}/list_attr_celeba.csv", newline="") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    self.header = row
                else:
                    filename = row[0]
                    self.filenames.append(filename)
                    self.annotations.append([int(v) for v in row[1:]])

        if limit is not None:
            self.annotations = np.array(self.annotations)
            self.filenames = self.filenames[:limit]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get the path to the image
        img_name = self.filenames[idx]
        img_path = os.path.join(self.dataset_folder, img_name)
        img_attributes = self.annotations[
            idx
        ]  # convert all attributes to zeros and ones
        # Load image and convert it to RGB
        img = Image.open(img_path).convert("RGB")
        # Apply transformations to the image
        if self.transform:
            img = self.transform(img)
        return img, {
            "filename": img_name,
            "idx": idx,
            "attributes": torch.tensor(img_attributes).long(),
        }


class CelebaCustomDataset(CelebADataset):
    def __getitem__(self, idx):
        indices = [8, 9, 11, 15, 16, 20, 22, 28, 35, 39]
        image, target = super().__getitem__(idx)
        target = target["attributes"] == 1
        new_target = target[indices]
        if sum(new_target) == 0:
            return self.__getitem__(np.random.randint(0, len(self)))
        return {"x": image, "y": new_target}


class ReferenceDataset(Dataset):
    def __init__(self, root, transform=None, limit=None):
        self.samples = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = sorted(os.listdir(root))
        samples = []
        for idx, domain in enumerate(domains):
            class_dir = os.path.join(root, domain)
            cls_fnames = os.listdir(class_dir)
            random_pairs = random.sample(cls_fnames, len(cls_fnames))
            for fname, fname2 in zip(cls_fnames, random_pairs):
                if fname == fname2:
                    fname2 = random_pairs[
                        (random_pairs.index(fname2) + 1) % len(cls_fnames)
                    ]
                samples.append(
                    (
                        os.path.join(class_dir, fname),
                        os.path.join(class_dir, fname2),
                        idx,
                    )
                )
        return samples

    def __getitem__(self, index):
        fname, fname2, label = self.samples[index]
        img = Image.open(fname).convert("RGB")
        img2 = Image.open(fname2).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)

        return {"ref1": img, "ref2": img2, "target": label}

    def __len__(self):
        return len(self.samples)
