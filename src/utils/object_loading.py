from operator import xor
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from src.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    transform = transforms.Compose(
        [
            transforms.Resize(configs.img_size),
            transforms.CenterCrop(configs.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == "train":
            drop_last = True
        else:
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(instantiate(ds, transform))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert "batch_size" in params, "You must provide batch_size for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(
            dataset
        ), f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            pin_memory=True,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        dataloaders[split] = dataloader
    return dataloaders
