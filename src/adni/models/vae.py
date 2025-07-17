import torch
from torch.nn import Module
from .base import Encoder, Decoder


class ConditionalVAE(Module):

    def __init__(self, num_classes, latent_dim=256, base_channels=32, backbone=None,
                 weights="DEFAULT", downsample_factor=4, bound_z_by=None):

        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(backbone=backbone, weights=weights, latent_dim=latent_dim,
                               base_channels=base_channels, downsample_factor=downsample_factor,
                               bound_z_by=bound_z_by)
        self.downsample_factor = self.encoder.downsample_factor
        self.upsample_factor = self.downsample_factor
        self.decoder = Decoder(latent_dim + num_classes, base_channels=base_channels, upsample_factor=self.upsample_factor)
        self.num_classes = num_classes

    def forward(self, x, y):
        z, mu, logvar = self.encoder(x)

        y = y.view(-1, self.num_classes, 1, 1)
        y = y.expand(-1, -1, z.shape[-2], z.shape[-1])

        z = torch.cat((z, y), dim=-3)  # z.shape: (num_classes + 256, 8, 8)

        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class InvariantVAE(Module):

    def __init__(self, cvae, latent_dim=256, base_channels=32, backbone=None,
                 weights="DEFAULT", downsample_factor=4, bound_z_by=None):
        super(InvariantVAE, self).__init__()

        self.encoder1 = cvae.encoder
        self.z1_dim = cvae.latent_dim
        for param in self.encoder1.parameters():
            param.requires_grad = False

        self.z2_dim = latent_dim
        self.encoder2 = Encoder(backbone=backbone, weights=weights, latent_dim=self.z2_dim,
                                base_channels=base_channels, downsample_factor=downsample_factor,
                                bound_z_by=bound_z_by)
        self.downsample_factor = self.encoder2.downsample_factor
        self.upsample_factor = self.downsample_factor
        self.decoder = Decoder(latent_dim=self.z1_dim + self.z2_dim,
                               base_channels=base_channels,
                               upsample_factor=self.upsample_factor)

    def forward(self, x):
        z1, _, _ = self.encoder1(x)
        z2, mu, logvar = self.encoder2(x)
        # Concatenate z1 and z2
        z = torch.cat((z1, z2), dim=-3)  # z.shape: (num_classes + 256, 8, 8)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
