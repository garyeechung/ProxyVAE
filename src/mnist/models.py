from typing import List

import torch
from torch.nn import Module, Linear, ReLU, Sequential, ModuleList


class CVAE(Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes=10):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, latent_dim * 2)  # Output mean and log variance
        )
        self.decoder = Sequential(
            Linear(latent_dim + num_classes, hidden_dim),  # +10 for the label
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        # Encode
        encoded = self.encoder(x)
        mu = encoded[:, :self.latent_dim]
        logvar = encoded[:, self.latent_dim:]

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Concatenate z with labels
        z_labels = torch.cat((z, labels), dim=1)

        # Decode
        reconstructed_x = self.decoder(z_labels)

        return reconstructed_x, mu, logvar


class InvariantAutoEncoder(Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cvae):
        super(InvariantAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cvae = cvae
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, latent_dim)  # Output mean and log variance
        )
        self.decoder = Sequential(
            Linear(latent_dim + self.cvae.latent_dim, hidden_dim),  # +10 for the label
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        with torch.no_grad():
            z_cvae = self.cvae.encoder(x)
            mu = z_cvae[:, :self.cvae.latent_dim]
            logvar = z_cvae[:, self.cvae.latent_dim:]
            z_cvae = self.cvae.reparameterize(mu, logvar)

        z = self.encoder(x)

        # Concatenate z with labels
        z = torch.cat((z, z_cvae), dim=-1)

        # Decode
        x_recon = self.decoder(z)

        return x_recon


class InvariantVariationalAutoEncoder(Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cvae):
        super(InvariantVariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cvae = cvae
        self.encoder = Sequential(
            Linear(input_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, latent_dim * 2)  # Output mean and log variance
        )
        self.decoder = Sequential(
            Linear(latent_dim + self.cvae.latent_dim, hidden_dim),  # +10 for the label
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        mu = encoded[:, :self.latent_dim]
        logvar = encoded[:, self.latent_dim:]

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        with torch.no_grad():
            z_cvae = self.cvae.encoder(x)
            mu_cvae = z_cvae[:, :self.cvae.latent_dim]
            logvar_cvae = z_cvae[:, self.cvae.latent_dim:]
            z_cvae = self.cvae.reparameterize(mu_cvae, logvar_cvae)

        # Concatenate z with labels
        z_combined = torch.cat((z, z_cvae), dim=1)

        # Decode
        reconstructed_x = self.decoder(z_combined)

        return reconstructed_x, mu, logvar


class ProxyRep2InvarRep(Module):
    def __init__(self, autoencoder, reparameterize: bool, hidden_layer_sizes: List[int] = [256, 512, 256]):
        super(ProxyRep2InvarRep, self).__init__()
        self.autoencoder = autoencoder
        self.reparameterize = reparameterize
        self.dim_proxy = autoencoder.latent_dim
        self.dim_invar = autoencoder.cvae.latent_dim

        self.mlp = ModuleList()
        self.mlp.append(Linear(self.dim_proxy, hidden_layer_sizes[0]))
        self.mlp.append(ReLU())
        for i in range(len(hidden_layer_sizes) - 1):
            self.mlp.append(Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))
            self.mlp.append(ReLU())
        self.mlp.append(Linear(hidden_layer_sizes[-1], self.dim_invar))
        self.mlp = Sequential(*self.mlp)

    def forward(self, x):
        # Encode proxy representation from x through invariant (V)AE
        with torch.no_grad():
            z_proxy = self.autoencoder.encoder(x)
        if self.reparameterize:
            mu = z_proxy[:, :self.dim_proxy]
            logvar = z_proxy[:, self.dim_proxy:]
            z_proxy = self.autoencoder.reparameterize(mu, logvar)

        # Pass through MLP to get invariant representation
        z_invar = self.mlp(z_proxy)
        return z_invar
