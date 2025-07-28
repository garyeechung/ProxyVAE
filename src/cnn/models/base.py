import torch
from torch.nn import Module, ModuleList, Sequential
from torch.nn import Conv2d, ConvTranspose2d, ReLU, Flatten
# from torchvision.models import resnet18, resnet50
import torchvision


class Encoder(Module):
    def __init__(self, backbone=None, weights="DEFAULT", latent_dim=256, base_channels=4,
                 downsample_factor=4, bound_z_by=None):
        super(Encoder, self).__init__()
        if backbone in ["resnet18", "resnet50"]:
            backbone = getattr(torchvision.models, backbone)(weights=weights)
            self.backbone = Sequential(*list(backbone.children())[:-2])
            self.downsample_factor = 5
            dummy_input = torch.zeros((1, 3, 128, 128))
            with torch.no_grad():
                z = self.backbone(dummy_input)
            backbone_out_channels = z.shape[1]

        else:
            self.backbone = ModuleList()
            self.downsample_factor = downsample_factor
            self.backbone.append(Conv2d(3, base_channels, 3, stride=1, padding=1))  # 128 -> 128
            self.backbone.append(ReLU())
            for i in range(downsample_factor):
                in_channels = base_channels * (2 ** i)
                out_channels = base_channels * (2 ** (i + 1))
                self.backbone.append(Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
                self.backbone.append(ReLU())
                self.backbone.append(Conv2d(out_channels, out_channels, 3, stride=1, padding=1))
                self.backbone.append(ReLU())
            self.backbone = Sequential(*self.backbone)

            backbone_out_channels = out_channels

        self.mu_enc = Conv2d(backbone_out_channels, latent_dim, 1)
        self.logvar_enc = Conv2d(backbone_out_channels, latent_dim, 1)
        self.flatten = Flatten(start_dim=1)  # Flatten the output to (batch_size, latent_dim)
        self.latent_dim = latent_dim
        if bound_z_by is not None:
            assert bound_z_by in ["tanh", "standardization", "normalization"], "Invalid output activation"
            if bound_z_by == "tanh":
                self.output_activation = torch.nn.Tanh()
            elif bound_z_by == "standardization":
                self.output_activation = Standardization()
            elif bound_z_by == "normalization":
                self.output_activation = Normalization()
        else:
            self.output_activation = None

    def reparameterize(self, mu, logvar):
        assert mu.shape == logvar.shape, "mu and logvar must have the same shape"
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_flattened=False):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        z = self.backbone(x)
        mu = self.mu_enc(z)
        logvar = self.logvar_enc(z)
        logvar = torch.clamp(logvar, -10, 10)
        if self.output_activation is not None:
            mu = self.output_activation(mu)
            logvar = self.output_activation(logvar)
        z = self.reparameterize(mu, logvar)
        if return_flattened:
            z = self.flatten(z)
            mu = self.flatten(mu)
            logvar = self.flatten(logvar)
        return z, mu, logvar


class Decoder(Module):
    def __init__(self, latent_dim=256, base_channels=4, upsample_factor=4, image_channels=1):
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

        self.conv.append(Conv2d(base_channels, image_channels, kernel_size=3, stride=1, padding=1))
        self.conv = Sequential(*self.conv)

    def forward(self, z):
        z = self.conv(z)
        return z


class Standardization(Module):
    def __init__(self, eps=1e-6):
        super(Standardization, self).__init__()
        self.eps = eps

    def forward(self, z):
        z_mean = z.mean(dim=(2, 3), keepdim=True)
        z_std = z.std(dim=(2, 3), keepdim=True)
        z = (z - z_mean) / (z_std + self.eps)
        return z


class Normalization(Module):
    def __init__(self, eps=1e-6):
        super(Normalization, self).__init__()
        self.eps = eps

    def forward(self, z):
        norm = torch.linalg.norm(z, dim=(2, 3), keepdim=True)
        z = z / (norm + self.eps)
        return z
