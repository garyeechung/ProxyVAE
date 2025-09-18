import torch
from torch.nn import Module
from .base import Encoder, Decoder


class ConditionalVAE(Module):

    def __init__(self, num_classes, latent_dim=16, hidden_dims=[1024, 512, 256],
                 input_dim=7260, bound_z_by=None):

        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim=input_dim, hidden_dims=hidden_dims,
                               latent_dim=latent_dim, bound_z_by=bound_z_by)
        self.decoder = Decoder(latent_dim + num_classes, hidden_dims=hidden_dims[::-1],
                               output_dim=input_dim)
        self.num_classes = num_classes

    def forward(self, x, y):
        z, mu, logvar = self.encoder(x)

        y = y.view(-1, self.num_classes)
        z = torch.cat((z, y), dim=-1)  # z.shape: (num_classes + 256,)

        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class ProxyVAE(Module):

    def __init__(self, cvae):
        super(ProxyVAE, self).__init__()

        self.encoder1 = cvae.encoder
        self.z1_dim = cvae.latent_dim
        self.z2_dim = cvae.latent_dim
        self.bound_z_by = cvae.encoder.bound_z_by
        for param in self.encoder1.parameters():
            param.requires_grad = False
        self.encoder2 = Encoder(input_dim=cvae.encoder.input_dim,
                                hidden_dims=cvae.encoder.hidden_dims,
                                latent_dim=self.z2_dim,
                                bound_z_by=cvae.encoder.bound_z_by)
        self.decoder = Decoder(latent_dim=self.z1_dim + self.z2_dim,
                               hidden_dims=cvae.decoder.hidden_dims,
                               output_dim=cvae.decoder.output_dim)

    def forward(self, x):
        z1, _, _ = self.encoder1(x)
        z2, mu, logvar = self.encoder2(x)
        # Concatenate z1 and z2
        z = torch.cat((z1, z2), dim=-1)  # z.shape: (num_classes + 256,)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
