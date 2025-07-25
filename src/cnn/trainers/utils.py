import base64
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import seaborn as sns
from sklearn import metrics


def vis_x_recon_comparison(x: Tensor, recon_x: Tensor, n=4, num_channels=1) -> np.ndarray:
    """
    Visualize the original and reconstructed images side by side.

    Args:
        x (Tensor): Original images, shape (B, C, H, W).
        recon_x (Tensor): Reconstructed images, shape (B, C, H, W).
        n (int): Number of images to visualize.
        num_channels (int): Number of channels (1 or 3).

    Returns:
        np.ndarray: Combined image of shape (2H, n*W, C) or (2H, n*W) for visualization.
    """
    x = x[:n]
    recon_x = recon_x[:n]

    # Remove NaNs/infs
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    recon_x = torch.nan_to_num(recon_x, nan=0.0, posinf=1.0, neginf=0.0)

    if num_channels == 1:
        # Shape: (n, 1, H, W) → (n, H, W)
        x = x.squeeze(1)
        recon_x = recon_x.squeeze(1)

        # Stack x and recon_x vertically, each is (n, H, W)
        img_rows = torch.cat([x, recon_x], dim=1)  # shape (n, 2H, W)

        # Join n samples horizontally: (2H, n*W)
        image_comparison = torch.cat([img for img in img_rows], dim=-1)
        image_comparison = torch.clip(image_comparison, 0., 1.).numpy()

        # Add black lines
        H, W = x.shape[1], x.shape[2]
        for i in range(1, n):
            image_comparison[:, i * W] = 1.0  # vertical line
        image_comparison[H, :] = 1.0  # horizontal line

    elif num_channels == 3:
        # Normalize per batch
        img_min = min(x.min(), recon_x.min())
        img_max = max(x.max(), recon_x.max())
        x = (x - img_min) / (img_max - img_min)
        recon_x = (recon_x - img_min) / (img_max - img_min)

        # Shape: (n, C, H, W) → (n, H, W, C)
        x = x.permute(0, 2, 3, 1)
        recon_x = recon_x.permute(0, 2, 3, 1)

        # Stack vertically then join horizontally
        img_rows = torch.cat([x, recon_x], dim=1)  # (n, 2H, W, C)
        image_comparison = torch.cat([img for img in img_rows], dim=1)  # (2H, n*W, C)
        image_comparison = image_comparison.numpy()

        # Optional: add vertical/horizontal lines (white)
        H, W = x.shape[1], x.shape[2]
        for i in range(1, n):
            image_comparison[:, i * W, :] = 1.0  # vertical lines
        image_comparison[H, :, :] = 1.0  # horizontal line

    else:
        raise ValueError("Unsupported number of channels. Only 1 or 3 supported.")

    return image_comparison


def convert_config_to_hash(config: dict, length: int = 8) -> str:
    # Convert to sorted JSON string for consistency
    json_str = json.dumps(config, sort_keys=True)
    sha1_digest = hashlib.sha1(json_str.encode()).digest()
    b64_encoded = base64.urlsafe_b64encode(sha1_digest).decode("utf-8")
    return b64_encoded[:length]


def get_confusion_matrix_heatmap_as_nparray(y_true: np.ndarray, y_pred: np.ndarray, annot=False) -> np.ndarray:

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=annot, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    return fig
