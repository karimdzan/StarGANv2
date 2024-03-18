import os
from pathlib import Path
import sys

import torch
import numpy as np

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils import prepare_device
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


def denormalize(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


@hydra.main(version_base=None, config_path="src/configs", config_name="test")
def main(config, test_dir, output_dir, device):
    logger = config.get_logger("test")
    OmegaConf.resolve(config)
    # define cpu or gpu if possible
    if config["device"] is None:
        device, device_ids = prepare_device(config["n_gpu"], logger)
        logger.info(f"Device {device} Ids {device_ids}")
    else:
        device = torch.device(config["device"])
        logger.info(f"Device {device}")

    # build model architecture
    model = instantiate(config["arch"])
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config["checkpoint"], map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    output_dir = config["output_dir"] if config["output_dir"] is not None else "outputs"
    os.makedirs(output_dir, exist_ok=True)
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)

    transform = transforms.Compose(
        [
            transforms.Resize(config["img_size"]),
            transforms.CenterCrop(config["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    with torch.no_grad():
        for src, ref in zip(os.listdir("pictures/src"), os.listdir("pictures/ref")):
            src_img = Image.open(os.path.join("pictures/src", src)).convert("RGB")
            src_img = transform(src_img).to(device)
            ref_img = Image.open(os.path.join("pictures/ref", ref)).convert("RGB")
            ref_img = transform(ref_img).to(device)

            domains = [8, 9, 11, 15, 16, 20, 22, 28, 35, 39]
            trg = torch.tensor(np.random.choice(domains)).to(device)
            s_ref = model.style_encoder(ref_img, trg)
            x_fake = model.generator(src_img, s_ref)

            x_fake = denormalize(x_fake)

            save_image(x_fake, os.path.join(output_dir, "fake.jpg"))


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
