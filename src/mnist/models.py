import torch
import torch.nn as nn
# import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_classes=10):
        super(CVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Output mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim),  # +10 for the label
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
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


class InvariantAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cvae):
        super(InvariantAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.cvae = cvae
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)  # Output mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.cvae.latent_dim, hidden_dim),  # +10 for the label
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        with torch.no_grad():
            z_cvae = self.cvae.encoder(x)
            mu = z_cvae[:, :self.cvae.latent_dim]
            logvar = z_cvae[:, self.cvae.latent_dim:]
            z_cvae = self.reparameterize(mu, logvar)

        z = self.encoder(x)

        # Concatenate z with labels
        z = torch.cat((z, z_cvae), dim=-1)

        # Decode
        x_recon = self.decoder(z)

        return x_recon


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.model(x)
