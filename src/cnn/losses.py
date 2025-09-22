import torch
import torch.nn as nn


class VAE_Loss(nn.Module):
    def __init__(self, beta):
        super(VAE_Loss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(reduction="mean")
        self.beta = beta

    def forward(self, x_recon, x, mu, logvar):
        summed_dims = [i for i in range(1, len(x.shape))]
        recon_loss = self.reconstruction_loss(x_recon, x)

        # Compute the KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=summed_dims).mean()

        # Combine the losses
        if self.beta <= 0:
            total_loss = recon_loss
        else:
            # If beta is positive, combine the losses
            # Otherwise, just return the reconstruction loss
            total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss


class VIB_Loss(nn.Module):
    def __init__(self, beta):
        super(VIB_Loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")
        self.beta = beta

    def forward(self, y_gt, y_pred, mu, logvar):
        summed_dims = [i for i in range(1, len(y_gt.shape))]
        # Compute the cross-entropy loss
        ce_loss = self.cross_entropy_loss(y_pred, y_gt)

        # Compute the KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=summed_dims).mean()

        # Combine the losses
        if self.beta <= 0:
            loss = ce_loss
        else:
            loss = ce_loss + self.beta * kl_loss

        return loss, ce_loss, kl_loss
