import colorsys
import io
from PIL import Image

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.manifold import TSNE


def convert_flatten_to_upper_triangle(x_flatten, d=121):
    assert x_flatten.shape[-1] == ((d * (d - 1)) // 2), "size of x_flatten"
    x_mat = np.zeros((d, d))
    x_mat[np.triu_indices(d, k=1)] = x_flatten
    return x_mat


def vis_x_recon_comparison(x, x_recon, d=121):
    x_mat = convert_flatten_to_upper_triangle(x, d)
    x_recon_mat = convert_flatten_to_upper_triangle(x_recon, d)
    comparison = np.concatenate([x_mat, x_recon_mat], axis=1)

    return comparison


def plot_tsne_adni(z: np.ndarray, yc: np.ndarray, yf: np.ndarray) -> Image:
    """
    z: (n, d): representations
    yc: (n,): coarse class labels, onehot encoded
    yf: (n,): fine class labels, onehot encoded
    """
    MERGE_GROUP = {
        0: [0, 1, 2, 3],  # GE
        1: [4, 5],        # Philips
        2: [6, 7, 8]     # Siemens
    }
    manu_mapping = {
        0: "GE",
        1: "Philips",
        2: "Siemens",
    }
    model_mapping = {
        0: "Discovery MR750",
        1: "Discovery MR750w",
        2: "Signa HDxt",
        3: "Signa Premier",
        4: "Achieva dStream",
        5: "Ingenia",
        6: "Prisma",
        7: "Prisma Fit",
        8: "Skyra"
    }

    coarse_base_rgb = {
        0: mcolors.to_rgb("orange"),
        1: mcolors.to_rgb("green"),
        2: mcolors.to_rgb("blue"),
    }

    pair_to_color = {}
    for coarse, base_rgb in coarse_base_rgb.items():
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        fine_in_group = MERGE_GROUP[coarse]
        n_fine = len(fine_in_group)
        lightness_values = np.linspace(0.2, 0.7, n_fine)

        for fine, lightness in zip(fine_in_group, lightness_values):
            rgb = colorsys.hls_to_rgb(h, lightness, s)
            pair_to_color[(coarse, fine)] = mcolors.to_hex(rgb)

    colors = [pair_to_color[(coarse, fine)] for coarse, fine in zip(yc, yf)]
    tsne = TSNE(n_components=2).fit_transform(z)
    comp_min, comp_max = tsne.min(0), tsne.max(0)
    tsne = (tsne - comp_min) / (comp_max - comp_min)

    fig, axes = plt.subplots(1, 1, figsize=(8, 5.5))
    axes.scatter(tsne[:, 0], tsne[:, 1], c=colors, s=50, alpha=0.9, edgecolor='none')
    axes.set_xticks([])
    axes.set_xticklabels([])
    axes.set_yticks([])
    axes.set_yticklabels([])
    axes.set_aspect('equal', adjustable='box')

    min_, max_ = tsne.min(), tsne.max()
    min_ -= 0.05 * (max_ - min_)
    max_ += 0.05 * (max_ - min_)
    axes.set_xlim(min_, max_)
    axes.set_ylim(min_, max_)

    handles = []
    for g, group in MERGE_GROUP.items():
        handles.append(Line2D([0], [0], color="none", label=manu_mapping[g], marker='o', linestyle='None'))
        for fine in group:
            handles.append(Line2D([0], [0],
                           color=pair_to_color[(g, fine)],
                           label=model_mapping[fine],
                           marker='o', linestyle='None',
                           alpha=0.9,
                           markersize=8))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.98, 0.5), fontsize=14, framealpha=0, handletextpad=0)
    plt.tight_layout(pad=0)
    plt.subplots_adjust(wspace=0.05, right=0.95)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig)
    return image
