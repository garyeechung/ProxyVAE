import torch
import torch.nn as nn


class VAE_Loss(nn.Module):
    def __init__(self, beta):
        super(VAE_Loss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction="mean")
        self.beta = beta

    def forward(self, x, x_recon, mu, logvar):
        # Compute the reconstruction loss
        recon_loss = self.reconstruction_loss(x_recon, x)

        # Compute the KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Combine the losses
        loss = recon_loss + self.beta * kl_loss

        return loss


class InfoBottleneck_Loss(nn.Module):
    def __init__(self, beta):
        super(InfoBottleneck_Loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")
        self.beta = beta

    def forward(self, y_gt, y_pred, mu, logvar):
        # Compute the cross-entropy loss
        ce_loss = self.cross_entropy_loss(y_pred, y_gt)

        # Compute the KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Combine the losses
        loss = ce_loss + self.beta * kl_loss
        return loss
