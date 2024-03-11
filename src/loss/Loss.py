import torch
from torch import nn
from src.loss import AdversarialLoss, GradientPenalty


class DiscriminatorLoss(nn.Module):
    def __init__(self, lambda_reg=10):
        super().__init__()
        self.adv_loss = AdversarialLoss()
        self.gradient_penalty = GradientPenalty()
        self.lambda_reg = lambda_reg

    def forward(
        self,
        model,
        x_real,
        y_org,
        y_trg,
        z_trg=None,
        x_ref=None,
    ):
        assert (z_trg is None) != (
            x_ref is None
        ), "Either z_trg or x_ref must be provided, not both or neither"
        x_real.requires_grad_()
        out_real = model.discriminator(x_real, y_org)
        loss_real = self.adv_loss(out_real, 1)
        loss_reg = self.gradient_penalty(out_real, x_real)

        # Handling fake images
        with torch.no_grad():
            if z_trg is not None:
                s_trg = model.mapping_network(z_trg, y_trg)
            else:
                s_trg = model.style_encoder(x_ref, y_trg)

            x_fake = model.generator(x_real, s_trg)
        out_fake = model.discriminator(x_fake, y_trg)
        loss_fake = self.adv_loss(out_fake, 0)

        loss = loss_real + loss_fake + self.lambda_reg * loss_reg
        return {
            "total": loss.item(),
            "real": loss_real.item(),
            "fake": loss_fake.item(),
            "reg": loss_reg.item(),
        }


class GeneratorLoss(nn.Module):
    def __init__(self, lambda_sty=0.5, lambda_ds=0.5, lambda_cyc=10):
        super().__init__()
        self.adv_loss = AdversarialLoss()
        self.lambda_sty = lambda_sty
        self.lambda_ds = lambda_ds
        self.lambda_cyc = lambda_cyc

    def forward(
        self,
        model,
        x_real,
        y_org,
        y_trg,
        z_trgs=None,
        x_refs=None,
    ):
        assert (z_trgs is None) != (
            x_refs is None
        ), "Either z_trgs or x_refs must be provided, not both or neither"
        if z_trgs is not None:
            z_trg, z_trg2 = z_trgs
        if x_refs is not None:
            x_ref, x_ref2 = x_refs

        # Initial style target
        s_trg = (
            model.mapping_network(z_trg, y_trg)
            if z_trgs is not None
            else model.style_encoder(x_ref, y_trg)
        )

        x_fake = model.generator(x_real, s_trg)
        out = model.discriminator(x_fake, y_trg)
        loss_adv = self.adv_loss(out, 1)

        s_pred = model.style_encoder(x_fake, y_trg)
        loss_sty = torch.mean(torch.abs(s_pred - s_trg))

        # Diversity sensitive and cycle-consistency losses
        s_trg2 = (
            model.mapping_network(z_trg2, y_trg)
            if z_trgs is not None
            else model.style_encoder(x_ref2, y_trg)
        )
        x_fake2 = model.generator(x_real, s_trg2).detach()
        loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

        s_org = model.style_encoder(x_real, y_org)
        x_rec = model.generator(x_fake, s_org)
        loss_cyc = torch.mean(torch.abs(x_rec - x_real))

        loss = (
            loss_adv
            + self.lambda_sty * loss_sty
            - self.lambda_ds * loss_ds
            + self.lambda_cyc * loss_cyc
        )
        return {
            "total": loss.item(),
            "adv": loss_adv.item(),
            "sty": loss_sty.item(),
            "ds": loss_ds.item(),
            "cyc": loss_cyc.item(),
        }
