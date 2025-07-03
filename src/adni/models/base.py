import torch
from torch.nn import Module, Sequential
from torch.nn import Conv2d, ConvTranspose2d, ReLU, Flatten


class Encoder(Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = Sequential(
            Conv2d(1, 32, 3, stride=1, padding=1),  # 128 -> 128
            ReLU(),
            Conv2d(32, 32, 3, stride=2, padding=1),  # 128 -> 64
            ReLU(),
            Conv2d(32, 64, 3, stride=1, padding=1),  # 64 -> 64
            ReLU(),
            Conv2d(64, 64, 3, stride=2, padding=1),  # 64 -> 32
            ReLU(),
            Conv2d(64, 128, 3, stride=1, padding=1),  # 32 -> 32
            ReLU(),
            Conv2d(128, 128, 3, stride=2, padding=1),  # 32 -> 16
            ReLU(),
            Conv2d(128, 256, 3, stride=1, padding=1),  # 16 -> 16
            ReLU(),
            Conv2d(256, 256, 3, stride=2, padding=1),  # 16 -> 8
            ReLU(),
            Conv2d(256, 256, 3, stride=1, padding=1),  # 8 -> 8
            ReLU(),
        )
        self.mu_enc = Conv2d(256, latent_dim, 1)
        self.logvar_enc = Conv2d(256, latent_dim, 1)
        self.flatten = Flatten(start_dim=1)  # Flatten the output to (batch_size, latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        assert mu.shape == logvar.shape, "mu and logvar must have the same shape"
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_flattened=False):
        z = self.conv(x)
        mu = self.mu_enc(z)
        logvar = self.logvar_enc(z)
        z = self.reparameterize(mu, logvar)
        if return_flattened:
            z = self.flatten(z)
            mu = self.flatten(mu)
            logvar = self.flatten(logvar)
        return z, mu, logvar


class Decoder(Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.conv = Sequential(
            Conv2d(latent_dim, 256, 1),
            ReLU(),
            Conv2d(256, 256, 3, stride=1, padding=1),  # 8 -> 8
            ReLU(),
            ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            ReLU(),
            Conv2d(256, 128, 3, stride=1, padding=1),  # 16 -> 16
            ReLU(),
            ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            ReLU(),
            Conv2d(128, 64, 3, stride=1, padding=1),  # 32 -> 32
            ReLU(),
            ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            ReLU(),
            Conv2d(64, 32, 3, stride=1, padding=1),  # 64 -> 64
            ReLU(),
            ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            ReLU(),
        )
        self.out = Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = self.conv(z)
        z = self.out(z)
        return z
