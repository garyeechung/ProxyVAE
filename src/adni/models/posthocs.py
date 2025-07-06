from typing import List, Tuple

import torch
from torch.nn import Module, Sequential, Linear, ReLU
from .base import Encoder
from .vae import InvariantVAE


class ProxyRep2InvaRep(Module):
    def __init__(self, ivae: InvariantVAE,
                 image_size: Tuple[int, int] = (224, 224)):
        super(ProxyRep2InvaRep, self).__init__()
        self.ivae = ivae
        for param in self.ivae.parameters():
            param.requires_grad = False

        # Assuming input size is divisible by 16
        # 16 because of 4 downsampling layers for the encoder
        bottleneck_size = (image_size[0] // 16) * (image_size[1] // 16)
        z1_dim = self.ivae.z1_dim * bottleneck_size
        z2_dim = self.ivae.z2_dim * bottleneck_size

        self.mlp = Sequential(
            Linear(z2_dim, 1024),
            ReLU(),
            Linear(1024, 2048),
            ReLU(),
            Linear(2048, 1024),
            ReLU(),
            Linear(1024, z1_dim),
            ReLU()
        )

    def forward(self, x):
        with torch.no_grad():
            z1, _, _ = self.ivae.encoder1(x, return_flattened=True)
            z2, _, _ = self.ivae.encoder2(x, return_flattened=True)
        return z1, z2, self.mlp(z2)


class VariationalPredictor(Module):
    def __init__(self, num_classes: int, is_posthoc: bool,
                 encoder: Encoder = None, latent_dim: int = 256,
                 image_size: List[int] = [224, 224]):
        super(VariationalPredictor, self).__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = Encoder(latent_dim=latent_dim)
            is_posthoc = False
        if is_posthoc:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.num_classes = num_classes
        self.flatten_dim = self.encoder.latent_dim * (image_size[0] // 16) * (image_size[1] // 16)

        self.mlp = Sequential(
            Linear(self.flatten_dim, 1024),
            ReLU(),
            Linear(1024, 2048),
            ReLU(),
            Linear(2048, 1024),
            ReLU(),
            Linear(1024, self.num_classes)
        )

    def forward(self, x):
        z, mu, logvar = self.encoder(x, return_flattened=True)
        return self.mlp(z), mu, logvar
