from abc import abstractmethod

import torch
from numpy import inf
import os
from glob import glob

from src.base import BaseModel
from src.logger import get_visualizer


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(
        self,
        model: BaseModel,
        criterion,
        metrics,
        optimizers,
        schedulers,
        config,
        device,
    ):
        self.device = device
        self.config = config
        self.logger = config.get_logger("trainer", config["trainer"]["verbosity"])

        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.optimizer_d = optimizers["discriminator"]
        self.optimizer_g = optimizers["generator"]
        self.optimizer_se = optimizers["style_encoder"]
        self.optimizer_mn = optimizers["mapping_network"]
        self.lr_scheduler_d = schedulers["discriminator"]
        self.lr_scheduler_g = schedulers["generator"]
        self.lr_scheduler_se = schedulers["style_encoder"]
        self.lr_scheduler_mn = schedulers["mapping_network"]

        # for interrupt saving
        self._last_epoch = 0

        cfg_trainer = config["trainer"]
        self.epochs = cfg_trainer["epochs"]
        self.save_period = cfg_trainer["save_period"]
        self.monitor = cfg_trainer.get("monitor", "off")
        self.reset_optimizer = cfg_trainer.get("reset_optimizer", False)
        self.reset_scheduler = cfg_trainer.get("reset_scheduler", False)

        # configuration to monitor model performance and save best
        if self.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = cfg_trainer.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = get_visualizer(config, self.logger, cfg_trainer["visualize"])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError()

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interrupt")
            self._save_checkpoint(self._last_epoch, save_best=False)
            raise e

    def _train_process(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._last_epoch = epoch
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {"epoch": epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info("    {:15s}: {}".format(str(key), value))

            # evaluate model performance according to configured metric,
            # save best checkpoint as model_best
            best = False
            if self.mnt_mode != "off":
                try:
                    # check whether model performance improved or not,
                    # according to specified metric(mnt_metric)
                    if self.mnt_mode == "min":
                        improved = log[self.mnt_metric] <= self.mnt_best
                    elif self.mnt_mode == "max":
                        improved = log[self.mnt_metric] >= self.mnt_best
                    else:
                        improved = False
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. "
                        "Model performance monitoring is disabled.".format(
                            self.mnt_metric
                        )
                    )
                    self.mnt_mode = "off"
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info(
                        "Validation performance didn't improve for {} epochs. "
                        "Training stops.".format(self.early_stop)
                    )
                    break

            if epoch % self.save_period == 0 or best:
                self._save_checkpoint(epoch, save_best=best, only_best=True)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__

        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_se": self.optimizer_se.state_dict(),
            "optimizer_mn": self.optimizer_mn.state_dict(),
            "lr_scheduler_d": self.lr_scheduler_d.state_dict(),
            "lr_scheduler_g": self.lr_scheduler_g.state_dict(),
            "lr_scheduler_se": self.lr_scheduler_se.state_dict(),
            "lr_scheduler_mn": self.lr_scheduler_mn.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / "checkpoint-epoch{}.pth".format(epoch))
        if not (only_best and save_best):
            for ckpt_path in glob(str(self.checkpoint_dir / "checkpoint-epoch*.pth")):
                os.remove(ckpt_path)
            torch.save(state, filename)
            self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        self.model.load_state_dict(checkpoint["state_dict"])

        if not self.reset_optimizer:
            self.logger.info("Loading optimizer state")
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])

        if not self.reset_scheduler:
            self.logger.info("Loading scheduler state")
            self.lr_scheduler_d.load_state_dict(checkpoint["lr_scheduler_d"])
            self.lr_scheduler_g.load_state_dict(checkpoint["lr_scheduler_g"])

        self.logger.info(
            "Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)
        )
