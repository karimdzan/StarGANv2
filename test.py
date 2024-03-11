import os
from pathlib import Path
import sys

import torch

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="src/configs", config_name="test")
def main(config, test_dir, output_dir, device):
    logger = config.get_logger("test")
    OmegaConf.resolve(config)
    # define cpu or gpu if possible
    device = torch.device(device)

    # build model architecture
    model = instantiate(config["arch"])
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    logger.info(f"Device {device}")
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)

    with torch.no_grad():
        


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
