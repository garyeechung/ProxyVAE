import torch
from torch.nn import Module, ModuleList, Sequential
from torch.nn import Conv2d, ConvTranspose2d, ReLU, Flatten
# from torchvision.models import resnet18, resnet50
import torchvision


class Encoder(Module):
    def __init__(self, backbone=None, weights="DEFAULT", latent_dim=256, base_channels=4, downsample_factor=4):
        super(Encoder, self).__init__()
        if backbone in ["resnet18", "resnet50"]:
            backbone = getattr(torchvision.models, backbone)(weights=weights)
            self.backbone = Sequential(*list(backbone.children())[:-2])
            self.downsample_factor = 5
            dummy_input = torch.zeros((1, 3, 128, 128))
            z = self.backbone(dummy_input)
            backbone_out_channels = z.shape[1]

        else:
            self.backbone = ModuleList()
            self.downsample_factor = downsample_factor
            self.backbone.append(Conv2d(1, base_channels, 3, stride=1, padding=1))  # 128 -> 128
            self.backbone.append(ReLU())

            for i in range(downsample_factor):
                in_channels = base_channels * (2 ** i)
                out_channels = base_channels * (2 ** (i + 1))
                self.backbone.append(Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
                self.backbone.append(ReLU())
                self.backbone.append(Conv2d(out_channels, out_channels, 3, stride=1, padding=1))
                self.backbone.append(ReLU())

            backbone_out_channels = out_channels
        self.mu_enc = Conv2d(backbone_out_channels, latent_dim, 1)
        self.logvar_enc = Conv2d(backbone_out_channels, latent_dim, 1)
        self.flatten = Flatten(start_dim=1)  # Flatten the output to (batch_size, latent_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        assert mu.shape == logvar.shape, "mu and logvar must have the same shape"
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_flattened=False):
        if self.backbone is not None:
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to RGB if needed
        z = self.backbone(x)
        mu = self.mu_enc(z)
        logvar = self.logvar_enc(z)
        z = self.reparameterize(mu, logvar)
        logvar = torch.clamp(logvar, -10, 10)
        if return_flattened:
            z = self.flatten(z)
            mu = self.flatten(mu)
            logvar = self.flatten(logvar)
        return z, mu, logvar


class Decoder(Module):
    def __init__(self, latent_dim=256, base_channels=4, upsample_factor=4):
        super(Decoder, self).__init__()
        self.conv = ModuleList()
        self.conv.append(Conv2d(latent_dim, base_channels * (2 ** upsample_factor), 1))
        self.conv.append(ReLU())
        for i in range(upsample_factor):
            in_channels = base_channels * (2 ** (upsample_factor - i))
            out_channels = base_channels * (2 ** (upsample_factor - i - 1))
            self.conv.append(Conv2d(in_channels, out_channels, 3, stride=1, padding=1))
            self.conv.append(ReLU())
            self.conv.append(ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=1, output_padding=1))
            self.conv.append(ReLU())

        self.conv.append(Conv2d(base_channels, 1, kernel_size=3, stride=1, padding=1))
        self.conv = Sequential(*self.conv)

    def forward(self, z):
        z = self.conv(z)
        return z
