import torch
import torch.nn as nn


class CVAE_Loss(nn.Module):
    def __init__(self, beta, reduction="sum"):
        super(CVAE_Loss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction=reduction)
        self.beta = beta

    def forward(self, x_recon, x, mu, logvar):
        # Compute the reconstruction loss
        recon_loss = self.reconstruction_loss(x_recon, x)

        # Compute the KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Combine the losses
        loss = recon_loss + self.beta * kl_loss

        return loss
