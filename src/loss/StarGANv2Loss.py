from torch import nn
from typing import Dict

from src.loss.Loss import DiscriminatorLoss, GeneratorLoss


class StarGANv2Loss(nn.Module):
    def __init__(self, generator_loss_config: Dict, discriminator_loss_config: Dict):
        super().__init__()
        self.discriminator_loss = DiscriminatorLoss(**discriminator_loss_config)
        self.generator_loss = GeneratorLoss(**generator_loss_config)
