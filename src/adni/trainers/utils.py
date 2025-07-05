import torch
from torch import Tensor


def vis_x_recon_comparison(x: Tensor, recon_x: Tensor, n=4):
    """
    Visualize the original and reconstructed images side by side.

    Args:
        x (Tensor): Original images tensor of shape (batch_size, channels, height, width).
        recon_x (Tensor): Reconstructed images tensor of shape (batch_size, channels, height, width).
        n (int): Number of images to visualize.
    """
    image_comparison = torch.cat([x, recon_x], dim=-2)
    image_comparison = torch.nan_to_num(image_comparison, nan=0.0, posinf=1.0, neginf=0.0)
    h, w = image_comparison.shape[-2:]
    image_comparison = torch.cat([images for images in image_comparison[:n]], dim=-1)
    image_comparison = torch.clip(image_comparison, 0., 1.)
    image_comparison = image_comparison.numpy()[0]

    for i in range(1, n):
        image_comparison[:, i * w] = 1.0
    image_comparison[h // 2, :] = 1.0

    return image_comparison
