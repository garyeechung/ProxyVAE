import colorsys
import io
from PIL import Image

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.manifold import TSNE


def plot_tsne(z: np.ndarray, yc: np.ndarray, yf: np.ndarray,
              merge_group,
              coarse_mapping: dict = None,
              fine_mapping: dict = None) -> Image:
    """
    z: (n, d): representations
    yc: (n,): coarse class labels, onehot encoded
    yf: (n,): fine class labels, onehot encoded
    """

    coarse_base_rgb = {
        0: mcolors.to_rgb("orange"),
        1: mcolors.to_rgb("green"),
        2: mcolors.to_rgb("blue"),
        3: mcolors.to_rgb("red"),
        4: mcolors.to_rgb("purple"),
        5: mcolors.to_rgb("brown"),
        6: mcolors.to_rgb("pink"),
        7: mcolors.to_rgb("gray"),
        8: mcolors.to_rgb("olive"),
        9: mcolors.to_rgb("cyan")
    }
    if coarse_mapping is None:
        num_coarse = len(merge_group)
        coarse_mapping = {i: i for i in range(num_coarse)}
    if fine_mapping is None:
        num_fine = sum([len(g) for g in merge_group.values()])
        fine_mapping = {i: i for i in range(num_fine)}
    coarse_base_rgb = {k: v for k, v in coarse_base_rgb.items() if k in merge_group}

    pair_to_color = {}
    for coarse, base_rgb in coarse_base_rgb.items():
        h, l, s = colorsys.rgb_to_hls(*base_rgb)
        fine_in_group = merge_group[coarse]
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
    for g, group in merge_group.items():
        handles.append(Line2D([0], [0], color="none", label=coarse_mapping[g], marker='o', linestyle='None'))
        for fine in group:
            handles.append(Line2D([0], [0],
                           color=pair_to_color[(g, fine)],
                           label=fine_mapping[fine],
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
