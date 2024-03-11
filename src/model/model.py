import torch
import torch.nn as nn
from typing import Tuple, Dict
import numpy as np

from src.model.base_model import BaseModel
from src.model.layers import AdaptiveResidualBlock, ResidualBlock


class Generator(BaseModel):
    def __init__(
        self,
        initial_dim: int = 64,
        img_size: int = 256,
        style_dim: int = 64,
        max_conv_dim: int = 512,
    ) -> None:
        super(Generator, self).__init__()
        self.from_rgb = nn.Conv2d(3, initial_dim, 3, 1, 1)
        self.encode, self.decode = self._build_blocks(
            initial_dim, img_size, style_dim, max_conv_dim
        )
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(initial_dim, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(initial_dim, 3, 1, 1, 0),
        )
        self.apply(init_conv_weight)

    def _build_blocks(
        self, dim_in: int, img_size: int, style_dim: int, max_conv_dim: int
    ) -> Tuple[nn.ModuleList, nn.ModuleList]:
        encode, decode = nn.ModuleList(), nn.ModuleList()
        repeat_num = int(np.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            encode.append(
                ResidualBlock(dim_in, dim_out, normalize=True, downsample=True)
            )
            decode.insert(
                0, AdaptiveResidualBlock(dim_out, dim_in, style_dim, upsample=True)
            )
            dim_in = dim_out
        for _ in range(2):
            encode.append(ResidualBlock(dim_out, dim_out, normalize=True))
            decode.insert(0, AdaptiveResidualBlock(dim_out, dim_out, style_dim))
        return encode, decode

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, style)
        return self.to_rgb(x)


class MappingNetwork(BaseModel):
    def __init__(
        self,
        latent_dim: int = 16,
        style_dim: int = 64,
        hidden_dim: int = 512,
        num_domains: int = 2,
    ) -> None:
        super(MappingNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            *(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) * 3,
        )

        self.domain_specific_layers = nn.ModuleList(
            [
                nn.Sequential(
                    *(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) * 3,
                    nn.Linear(hidden_dim, style_dim),
                )
                for _ in range(num_domains)
            ]
        )

        self.apply(init_conv_weight)
        self.apply(init_fc_weight_zero)

    def forward(
        self, latent_code: torch.Tensor, domain_indices: torch.Tensor
    ) -> torch.Tensor:
        shared_output = self.shared_layers(latent_code)

        style_codes = [
            domain_layer(shared_output) for domain_layer in self.domain_specific_layers
        ]
        style_codes = torch.stack(style_codes, dim=1)

        selected_style_codes = style_codes[
            torch.arange(domain_indices.size(0)), domain_indices
        ]

        return selected_style_codes


class StyleEncoder(BaseModel):
    def __init__(
        self,
        initial_dim: int = 64,
        img_size: int = 256,
        style_dim: int = 64,
        num_domains: int = 10,
        max_conv_dim: int = 512,
    ) -> None:
        super(StyleEncoder, self).__init__()
        self.features, final_dim = self._build_convolutional_blocks(
            initial_dim, img_size, max_conv_dim
        )

        self.flatten = nn.Flatten()

        self.style_projections = nn.ModuleList(
            [nn.Linear(final_dim, style_dim) for _ in range(num_domains)]
        )

        self.apply(init_conv_weight)

    def _build_convolutional_blocks(
        self, initial_dim: int, img_size: int, max_conv_dim: int
    ) -> Tuple[nn.Sequential, int]:
        blocks = [nn.Conv2d(3, initial_dim, kernel_size=3, stride=1, padding=1)]
        dim_in = initial_dim

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResidualBlock(dim_in, dim_out, downsample=True))
            dim_in = dim_out

        blocks.extend(
            [
                nn.LeakyReLU(0.2),
                nn.Conv2d(dim_out, dim_out, kernel_size=4, stride=1, padding=0),
                nn.LeakyReLU(0.2),
            ]
        )

        return nn.Sequential(*blocks), dim_out

    def forward(self, x: torch.Tensor, domain_indices: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)

        style_codes = [projection(x) for projection in self.style_projections]
        style_codes = torch.stack(style_codes, dim=1)

        selected_style_codes = style_codes[
            torch.arange(domain_indices.size(0)), domain_indices
        ]

        return selected_style_codes


class Discriminator(BaseModel):
    def __init__(
        self,
        initial_dim: int = 64,
        img_size: int = 256,
        num_domains: int = 10,
        max_conv_dim: int = 512,
    ) -> None:
        super(Discriminator, self).__init__()

        self.blocks = nn.Sequential(
            nn.Conv2d(3, initial_dim, kernel_size=3, stride=1, padding=1),
            *self._build_downsampling_blocks(initial_dim, img_size, max_conv_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(max_conv_dim, max_conv_dim, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(max_conv_dim, num_domains, kernel_size=1, stride=1, padding=0),
        )

        self.flatten = nn.Flatten()

        self.apply(init_conv_weight)

    def _build_downsampling_blocks(
        self, initial_dim: int, img_size: int, max_conv_dim: int
    ) -> list:
        blocks = []
        dim_in = initial_dim
        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks.append(ResidualBlock(dim_in, dim_out, downsample=True))
            dim_in = dim_out
        return blocks

    def forward(self, x: torch.Tensor, domain_indices: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = self.flatten(x)  # (batch, num_domains)
        selected_style_codes = x[torch.arange(domain_indices.size(0)), domain_indices]
        return selected_style_codes


class StarGANv2(BaseModel):
    def __init__(
        self,
        generator_config: Dict,
        mapping_network_config: Dict,
        style_encoder_config: Dict,
        discriminator_config: Dict,
    ) -> None:
        super(StarGANv2, self).__init__()
        self.generator = Generator(**generator_config)
        self.mapping_network = MappingNetwork(**mapping_network_config)
        self.style_encoder = StyleEncoder(**style_encoder_config)
        self.discriminator = Discriminator(**discriminator_config)


def init_conv_weight(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def init_fc_weight_zero(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(m.bias, 0.0)
