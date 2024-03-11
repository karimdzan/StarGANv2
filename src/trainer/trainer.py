import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.trainer.base_trainer import BaseTrainer
from src.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
        self,
        model,
        criterion,
        metrics,
        optimizers,
        schedulers,
        config,
        device,
        dataloaders,
        len_epoch=None,
        skip_oom=True,
    ):
        super().__init__(
            model,
            criterion,
            metrics,
            optimizers,
            schedulers,
            config,
            device,
        )
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        self.ref_dataloader = dataloaders["ref"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {
            k: v for k, v in dataloaders.items() if k != "train"
        }
        self.log_step = self.config["trainer"]["log_step"]
        self.eval_step = self.config["trainer"]["eval_step"]
        self.latent_dim = self.config["trainer"]["latent_dim"]

        self.train_metrics = MetricTracker(
            "gen_latent",
            "disc_latent",
            "gen_ref",
            "disc_ref",
            "sty",
            "cyc",
            writer=self.writer,
        )
        self.evaluation_metrics = MetricTracker(
            *[m.name for m in self.metrics], writer=self.writer
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["x", "y"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    @staticmethod
    def move_batch_to_device_ref(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["ref1", "ref2", "target"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self, module):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                module.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()

        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        dataloader_iterator = iter(self.ref_dataloader)

        for batch_idx, batch in enumerate(
            tqdm(
                self.train_dataloader,
                desc="train",
                total=self.len_epoch,
            )
        ):
            try:
                batch_ref = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(self.ref_dataloader)
                batch_ref = next(dataloader_iterator)
            try:
                batch = self.process_batch(
                    batch,
                    batch_ref,
                    batch_idx,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} DLoss_latent: {:.6f} GLoss_latent: {:.6f} DLoss_ref: {:.6f} GLoss_ref: {:.6f} SELoss {:.6f} MNLoss {:.6f}".format(
                        epoch,
                        self._progress(batch_idx),
                        batch["gen_latent"],
                        batch["disc_latent"],
                        batch["gen_ref"],
                        batch["disc_ref"],
                        batch["cyc"],
                        batch["sty"],
                    )
                )
                self.writer.add_scalar(
                    "discriminator learning rate", self.lr_scheduler_d.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "generator learning rate", self.lr_scheduler_g.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "style encoder learning rate", self.lr_scheduler_se.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "mapping network learning rate",
                    self.lr_scheduler_mn.get_last_lr()[0],
                )
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx % self.eval_step == 0:
                s_ref = self.model.style_encoder(batch_ref["ref1"], batch_ref["target"])
                x_fake = self.model.generator(batch["x"], s_ref)
                s_src = self.model.style_encoder(batch["x"], batch["y"])
                x_rec = self.model.generator(x_fake, s_src)

                x = self.denormalize(batch["x"])
                ref = self.denormalize(batch_ref["ref1"])
                x_fake = self.denormalize(x_fake)
                x_rec = self.denormalize(x_rec)
                self.writer.add_table_images(x, ref, x_fake, x_rec)
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        self.lr_scheduler_d.step()
        self.lr_scheduler_g.step()
        self.lr_scheduler_se.step()
        self.lr_scheduler_mn.step()

        return log

    def process_batch(self, batch, batch_ref, batch_idx, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        batch_ref = self.move_batch_to_device_ref(batch_ref, self.device)

        self._clip_grad_norm(self.model.discriminator)

        x_real, y_org = batch["x"], batch["y"]
        x_ref, x_ref2, y_trg = batch_ref["ref1"], batch_ref["ref2"], batch_ref["target"]
        z_trg = torch.randn(x_real.size(0), self.latent_dim).to(self.device)
        z_trg2 = torch.randn(x_real.size(0), self.latent_dim).to(self.device)

        # train the discriminator
        d_loss = self.criterion.discriminator_loss(
            self.model, x_real, y_org, y_trg, z_trg=z_trg
        )
        self._reset_grad()
        d_loss["total"].backward()
        self.optimizer_d.step()
        batch["disc_latent"] = d_loss["total"].item()

        d_loss = self.criterion.discriminator_loss(
            self.model, x_real, y_org, y_trg, x_ref=x_ref
        )
        self._reset_grad()
        d_loss["total"].backward()
        self.optimizer_d.step()
        batch["disc_ref"] = d_loss["total"].item()

        # train the generator
        g_loss = self.criterion.generator_loss(
            self.model, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2]
        )
        self._reset_grad()
        g_loss["total"].backward()
        self.optimizer_g.step()
        self.optimizer_mn.step()
        self.optimizer_se.step()

        batch["gen_latent"] = g_loss["total"].item()
        batch["cyc"] = g_loss["cyc"]
        batch["sty"] = g_loss["sty"]
        g_loss = self.criterion.generator_loss(
            self.model, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2]
        )
        self._reset_grad()
        g_loss["total"].backward()
        self.optimizer_g.step()

        batch["gen_ref"] = g_loss["total"].item()

        self._clip_grad_norm(self.model.generator)

        for metric_key in metrics.keys():
            metrics.update(metric_key, batch[metric_key])

        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @torch.no_grad()
    def get_grad_norm(self, parameters, norm_type=2):
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]

        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

    def _reset_grad(self):
        self.optimizer_d.zero_grad()
        self.optimizer_g.zero_grad()
        self.optimizer_se.zero_grad()
        self.optimizer_mn.zero_grad()

    def denormalize(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)
