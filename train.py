import warnings

import numpy as np
import torch

import logging
import sys
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config: DictConfig):
    logger = logging.getLogger("train")

    OmegaConf.resolve(config)
    config = ConfigParser(OmegaConf.to_container(config))

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    model = instantiate(config["StarGANv2"])
    logger.info(model.generator)
    logger.info(model.discriminator)
    logger.info(model.mapping_network)
    logger.info(model.style_encoder)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"], logger)
    logger.info(f"Device {device} Ids {device_ids}")
    model = model.to(device)

    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    loss_module = instantiate(config["loss"]).to(device)
    metrics = [instantiate(metric_dict) for metric_dict in config["metrics"]]

    # logger.info(f'Len epoch {config["trainer"]["len_epoch"]}')
    logger.info(f'Epochs {config["trainer"]["epochs"]}')
    logger.info(f'Dataset size {len(dataloaders["train"].dataset)}')
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params_d = filter(
        lambda p: p.requires_grad,
        model.discriminator.parameters(),
    )
    optimizer_d = instantiate(config["optimizer_d"], trainable_params_d)
    trainable_params_g = filter(lambda p: p.requires_grad, model.generator.parameters())
    optimizer_g = instantiate(config["optimizer_g"], trainable_params_g)
    trainable_params_se = filter(
        lambda p: p.requires_grad, model.style_encoder.parameters()
    )
    optimizer_se = instantiate(config["optimizer_se"], trainable_params_se)
    trainable_params_mn = filter(
        lambda p: p.requires_grad, model.mapping_network.parameters()
    )
    optimizer_mn = instantiate(config["optimizer_mn"], trainable_params_mn)

    lr_scheduler_d = instantiate(config["lr_scheduler_d"], optimizer_d)
    lr_scheduler_g = instantiate(config["lr_scheduler_g"], optimizer_g)
    lr_scheduler_se = instantiate(config["lr_scheduler_se"], optimizer_se)
    lr_scheduler_mn = instantiate(config["lr_scheduler_mn"], optimizer_mn)

    optimizers = {
        "generator": optimizer_g,
        "discriminator": optimizer_d,
        "style_encoder": optimizer_se,
        "mapping_network": optimizer_mn,
    }

    lr_schedulers = {
        "generator": lr_scheduler_g,
        "discriminator": lr_scheduler_d,
        "style_encoder": lr_scheduler_se,
        "mapping_network": lr_scheduler_mn,
    }

    trainer = Trainer(
        model,
        loss_module,
        metrics,
        optimizers,
        lr_schedulers,
        config=config,
        device=device,
        dataloaders=dataloaders,
        len_epoch=config["trainer"].get("len_epoch", None),
    )

    trainer.train()


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
