from typing import Tuple

from torch.nn import Module, Sequential, Linear, ReLU
from .base import Encoder
from .vae import InvariantVAE


class ProxyRep2InvarRep(Module):
    def __init__(self, ivae: InvariantVAE,
                 image_size: Tuple[int, int] = (128, 128)):
        super(ProxyRep2InvarRep, self).__init__()
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

    def forward(self, z2):
        # assuming z2 is flattened to (batch_size, z2_dim)
        return self.mlp(z2)


class VariationalPredictor(Module):
    def __init__(self, encoder: Encoder):
        super(VariationalPredictor, self).__init__()
        self.encoder = encoder

    def forward(self, x):
        return None
