import os
import torch
from natsort import natsorted
from PIL import Image
from torch.utils.data import Dataset
import csv
import numpy as np
import random
from src.datasets.base_dataset import BaseDataset
from itertools import compress


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

        self.annotations = np.array(self.annotations)
        if limit is not None:
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        indices = [8, 9, 11, 15, 16, 20, 22, 28, 35, 39]
        image, target = super().__getitem__(idx)
        target = target["attributes"] == 1
        new_target = target[indices]
        if sum(new_target) == 0:
            return self.__getitem__(np.random.randint(0, len(self)))

        nonzero_indices = torch.nonzero(new_target, as_tuple=True)[0]
        shuffled_indices = torch.randperm(nonzero_indices.size(0))
        return {"x": image, "y": nonzero_indices[shuffled_indices[0]]}


class ReferenceDataset(CelebADataset):
    def __init__(self, root_dir, transform=None, limit=None):
        super().__init__(root_dir, transform, limit)
        self.samples, self.targets = self._make_dataset(root_dir)
        if limit is not None:
            self.samples = self.samples[:limit]
            self.targets = self.targets[:limit]
        self.transform = transform
        self.root_dir = root_dir

    def _make_dataset(self, root_dir):
        domains = [8, 9, 11, 15, 16, 20, 22, 28, 35, 39]
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(domains):
            cls_fnames = list(
                compress(self.filenames, self.annotations[:, domain] == 1)
            )
            fnames += cls_fnames
            fnames2 += random.sample(cls_fnames, len(cls_fnames))
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(
            os.path.join(self.root_dir, "img_align_celeba/" + fname)
        ).convert("RGB")
        img2 = Image.open(
            os.path.join(self.root_dir, "img_align_celeba/" + fname2)
        ).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return {"ref1": img, "ref2": img2, "target": label}

    def __len__(self):
        return len(self.targets)
