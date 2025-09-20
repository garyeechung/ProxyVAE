from typing import List

import torch
from torch.nn import Module, Sequential, ModuleList, Linear, ReLU, Softmax
from .base import Encoder
from .vae import ProxyVAE


class ProxyRep2InvaRep(Module):
    def __init__(self, proxyvae: ProxyVAE,
                 hidden_dims: List[int] = [2048, 2048, 2048]):
        super(ProxyRep2InvaRep, self).__init__()
        self.proxyvae = proxyvae
        for param in self.proxyvae.parameters():
            param.requires_grad = False
        device = next(self.proxyvae.parameters()).device

        layers = ModuleList()
        layers.append(Linear(self.proxyvae.z2_dim, hidden_dims[0]))
        layers.append(ReLU())
        for i in range(1, len(hidden_dims)):
            layers.append(Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(ReLU())
        layers.append(Linear(hidden_dims[-1], self.proxyvae.z1_dim))
        self.mlp = Sequential(*layers).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z1, _, _ = self.proxyvae.encoder1(x)
            z2, _, _ = self.proxyvae.encoder2(x)
        return z1, z2, self.mlp(z2)


class VariationalPredictor(Module):
    def __init__(self, num_classes: int, is_posthoc: bool,
                 encoder: Encoder = None, latent_dim: int = 16,
                 hidden_dims: List[int] = [1024, 512, 256],
                 bound_z_by: str = None):
        super(VariationalPredictor, self).__init__()
        if encoder is not None:
            self.encoder = encoder
            self.latent_dim = self.encoder.latent_dim
        else:
            self.encoder = Encoder(latent_dim=latent_dim, hidden_dims=hidden_dims, bound_z_by=bound_z_by)
            self.latent_dim = latent_dim
            is_posthoc = False
        if is_posthoc:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.num_classes = num_classes
        self.classifier = ModuleList()
        prev_dim = self.latent_dim
        for h_dim in [32, 32, 32]:
            self.classifier.append(Linear(prev_dim, h_dim))
            self.classifier.append(ReLU())
            prev_dim = h_dim
        self.classifier.append(Linear(prev_dim, num_classes))
        self.classifier.append(Softmax(dim=-1))
        self.classifier = Sequential(*self.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mu, logvar = self.encoder(x)
        y_pred = self.classifier(z)
        return y_pred, mu, logvar
