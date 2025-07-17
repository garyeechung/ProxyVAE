import base64
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import seaborn as sns
from sklearn import metrics


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


def convert_config_to_hash(config: dict, length: int = 8) -> str:
    # Convert to sorted JSON string for consistency
    json_str = json.dumps(config, sort_keys=True)
    sha1_digest = hashlib.sha1(json_str.encode()).digest()
    b64_encoded = base64.urlsafe_b64encode(sha1_digest).decode("utf-8")
    return b64_encoded[:length]


def get_confusion_matrix_heatmap_as_nparray(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    return fig
