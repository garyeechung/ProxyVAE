from typing import List, Tuple

import torch
from torch.nn import Module, Sequential, Linear, ReLU, Softmax
from .base import Encoder
from .vae import InvariantVAE


class ProxyRep2InvaRep(Module):
    def __init__(self, ivae: InvariantVAE, image_size: Tuple[int, int] = (224, 224)):
        super(ProxyRep2InvaRep, self).__init__()
        self.ivae = ivae
        for param in self.ivae.parameters():
            param.requires_grad = False

        mock_image = torch.zeros((1, 1, *image_size))
        with torch.no_grad():
            mock_z1, _, _ = self.ivae.encoder1(mock_image, return_flattened=True)
            mock_z2, _, _ = self.ivae.encoder2(mock_image, return_flattened=True)

        z1_dim = mock_z1.shape[-1]
        z2_dim = mock_z2.shape[-1]

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
    """ A classifier that predicts the class of the input image
    """
    def __init__(self, num_classes: int, is_posthoc: bool,
                 encoder: Encoder = None, latent_dim: int = 256,
                 base_channels: int = 4, image_size: List[int] = [224, 224],
                 image_channels: int = 1, backbone: str = None, weights: str = "DEFAULT"):
        super(VariationalPredictor, self).__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = Encoder(backbone=backbone, weights=weights, latent_dim=latent_dim)
            is_posthoc = False
        if is_posthoc:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.num_classes = num_classes
        mock_image = torch.zeros((1, image_channels, *image_size))
        mock_z, _, _ = self.encoder(mock_image, return_flattened=True)
        self.flatten_dim = mock_z.shape[-1]

        self.mlp = Sequential(
            Linear(self.flatten_dim, base_channels * 32),
            ReLU(),
            Linear(base_channels * 32, base_channels * 64),
            ReLU(),
            Linear(base_channels * 64, base_channels * 32),
            ReLU(),
            Linear(base_channels * 32, self.num_classes),
            Softmax(dim=-1)
        )

    def forward(self, x):
        z, mu, logvar = self.encoder(x, return_flattened=True)
        y_pred = self.mlp(z)
        return y_pred, mu, logvar
