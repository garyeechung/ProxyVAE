import torch
from torch.nn import Module, ModuleList, Sequential, ReLU


class Encoder(Module):
    def __init__(self, input_dim=7260, hidden_dims=[1024, 512, 256],
                 latent_dim=16, bound_z_by=None):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        layers = ModuleList()
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(ReLU())
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, latent_dim * 2))
        self.bound_z_by = bound_z_by
        if self.bound_z_by is not None:
            assert self.bound_z_by in ["tanh", "standardization", "normalization"], "Invalid output activation"
            if self.bound_z_by == "tanh":
                layers.append(torch.nn.Tanh())
            elif self.bound_z_by == "standardization":
                layers.append(Standardization())
            elif self.bound_z_by == "normalization":
                layers.append(Normalization())
        self.encoder = Sequential(*layers)

    def reparameterize(self, mu, logvar):
        assert mu.shape == logvar.shape, "mu and logvar must have the same shape"
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z = self.encoder(x)
        mu, logvar = z[:, :self.latent_dim], z[:, self.latent_dim:]
        logvar = torch.clamp(logvar, -10, 10)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(Module):
    def __init__(self, latent_dim=16, hidden_dims=[256, 512, 1024],
                 output_dim=7260):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        layers = ModuleList()
        prev_dim = latent_dim
        for h_dim in self.hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(ReLU())
            prev_dim = h_dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        self.decoder = Sequential(*layers)

    def forward(self, z):
        x_recon = self.decoder(z)
        return x_recon


class Standardization(Module):
    def __init__(self, eps=1e-6):
        super(Standardization, self).__init__()
        self.eps = eps

    def forward(self, z):
        z_mean = z.mean(dim=1, keepdim=True)
        z_std = z.std(dim=1, keepdim=True)
        z = (z - z_mean) / (z_std + self.eps)
        return z


class Normalization(Module):
    def __init__(self, eps=1e-6):
        super(Normalization, self).__init__()
        self.eps = eps

    def forward(self, z):
        norm = torch.linalg.norm(z, dim=1, keepdim=True)
        z = z / (norm + self.eps)
        return z
