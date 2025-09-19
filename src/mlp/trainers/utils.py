import numpy as np


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
