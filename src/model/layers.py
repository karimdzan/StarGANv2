import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: nn.Module = nn.LeakyReLU(0.2),
        normalize: bool = False,
        downsample: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.normalize = normalize
        self.downsample = downsample
        self.learned_shortcut = input_dim != output_dim
        self.conv_layers = nn.Sequential(
            nn.InstanceNorm2d(output_dim, affine=True) if normalize else nn.Identity(),
            self.activation,
            nn.Conv2d(input_dim, output_dim, 3, stride=1, padding=1),
            nn.AvgPool2d(2) if self.downsample else nn.Identity(),
            nn.InstanceNorm2d(output_dim, affine=True) if normalize else nn.Identity(),
            self.activation,
            nn.Conv2d(output_dim, output_dim, 3, stride=1, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 1, stride=1, padding=0, bias=False)
            if self.learned_shortcut
            else nn.Identity(),
            nn.AvgPool2d(2) if self.downsample else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.shortcut(x) + self.conv_layers(x)) / np.sqrt(2)


class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self, style_dim: int, num_features: int):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_scale_shift = nn.Linear(style_dim, num_features * 2)
        self.apply(init_fc_weight_one)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        style = self.style_scale_shift(style).view(style.size(0), -1, 1, 1)
        gamma, beta = style.chunk(2, 1)
        return (1 + gamma) * self.instance_norm(x) + beta


class AdaptiveResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        style_dim: int,
        activation: nn.Module = nn.LeakyReLU(0.2),
        upsample: bool = False,
    ):
        super().__init__()
        self.learned_shortcut = input_dim != output_dim

        if self.learned_shortcut:
            layers = [
                nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=1, padding=0)
            ]
            if upsample:
                layers.insert(0, nn.Upsample(scale_factor=2, mode="nearest"))
            self.shortcut = nn.Sequential(*layers)
        else:
            self.shortcut = (
                None if not upsample else nn.Upsample(scale_factor=2, mode="nearest")
            )

        self.main = nn.Sequential(
            AdaptiveInstanceNormalization(style_dim, input_dim),
            activation,
            nn.Upsample(scale_factor=2, mode="nearest") if upsample else nn.Identity(),
            nn.Conv2d(input_dim, output_dim, 3, 1, 1),
            AdaptiveInstanceNormalization(style_dim, output_dim),
            activation,
            nn.Conv2d(output_dim, output_dim, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x) if self.shortcut is not None else x

        for layer in self.main:
            if isinstance(layer, AdaptiveInstanceNormalization):
                x = layer(x, style)
            else:
                x = layer(x)

        return (shortcut + x) / np.sqrt(2)


def init_fc_weight_one(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(m.bias, 1.0)
