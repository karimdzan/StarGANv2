import torch
from torch import nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, target):
        assert target in [1, 0], "Target must be 1 (real) or 0 (fake)"
        targets = torch.full_like(logits, fill_value=target)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss


class GradientPenalty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(),
            inputs=x_in,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert grad_dout2.size() == x_in.size(), "Mismatch in gradient and input sizes"
        reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
        return reg
