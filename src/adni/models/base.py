import torch
from torch.nn import Module, Sequential
from torch.nn import Conv2d, ConvTranspose2d, ReLU, Flatten
from torchvision.models import resnet18


class Encoder(Module):
    def __init__(self, conv=None, weights="DEFAULT", latent_dim=256, base_channels=32):
        super().__init__()
        if conv == "resnet18":
            resnet = resnet18(weights=weights)
            self.conv = Sequential(*list(resnet.children())[:-2])
            dummy_input = torch.zeros((1, 3, 128, 128))
            z = self.conv(dummy_input)
            conv_out_channels = z.shape[1]

        else:
            self.conv = Sequential(
                Conv2d(1, base_channels, 3, stride=1, padding=1),  # 128 -> 128
                ReLU(),
                Conv2d(base_channels, base_channels, 3, stride=2, padding=1),  # 128 -> 64
                ReLU(),
                Conv2d(base_channels, base_channels * 2, 3, stride=1, padding=1),  # 64 -> 64
                ReLU(),
                Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1),  # 64 -> 32
                ReLU(),
                Conv2d(base_channels * 2, base_channels * 4, 3, stride=1, padding=1),  # 32 -> 32
                ReLU(),
                Conv2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1),  # 32 -> 16
                ReLU(),
                Conv2d(base_channels * 4, base_channels * 8, 3, stride=1, padding=1),  # 16 -> 16
                ReLU(),
                Conv2d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1),  # 16 -> 8
                ReLU(),
                Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, padding=1),  # 8 -> 8
                ReLU(),
            )
            conv_out_channels = base_channels * 8
        self.mu_enc = Conv2d(conv_out_channels, latent_dim, 1)
        self.logvar_enc = Conv2d(conv_out_channels, latent_dim, 1)
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
    def __init__(self, latent_dim=256, base_channels=32):
        super().__init__()
        self.conv = Sequential(
            Conv2d(latent_dim, base_channels * 8, 1),
            ReLU(),
            Conv2d(base_channels * 8, base_channels * 8, 3, stride=1, padding=1),  # 8 -> 8
            ReLU(),
            ConvTranspose2d(base_channels * 8, base_channels * 8, 3, stride=2, padding=1, output_padding=1),  # 8 -> 16
            ReLU(),
            Conv2d(base_channels * 8, base_channels * 4, 3, stride=1, padding=1),  # 16 -> 16
            ReLU(),
            ConvTranspose2d(base_channels * 4, base_channels * 4, 3, stride=2, padding=1, output_padding=1),  # 16 -> 32
            ReLU(),
            Conv2d(base_channels * 4, base_channels * 2, 3, stride=1, padding=1),  # 32 -> 32
            ReLU(),
            ConvTranspose2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1, output_padding=1),  # 32 -> 64
            ReLU(),
            Conv2d(base_channels * 2, base_channels, 3, stride=1, padding=1),  # 64 -> 64
            ReLU(),
            ConvTranspose2d(base_channels, base_channels, 3, stride=2, padding=1, output_padding=1),  # 64 -> 128
            ReLU(),
        )
        self.out = Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = self.conv(z)
        z = self.out(z)
        return z
