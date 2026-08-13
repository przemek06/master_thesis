import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from generate import generate_sparse_householder

HERE = os.path.dirname(__file__)

N = 200
SPARSITY = 0.9
SEED = 0

SPECTRA = [
    ("big ring", 0.85, 0.90),
    ("small ring", 0.25, 0.3)
]


def sample_ring(r_min, r_max, size):
    radii = np.sqrt(np.random.uniform(r_min ** 2, r_max ** 2, size))
    angles = np.random.uniform(0, 2 * np.pi, size)
    return radii * np.exp(1j * angles)


def block_diagonal(eigs, n):
    D = np.zeros((n, n))
    for i, lam in enumerate(eigs):
        a, b = lam.real, lam.imag
        D[2 * i:2 * i + 2, 2 * i:2 * i + 2] = [[a, -b], [b, a]]
    return D


def build(r_min, r_max):
    np.random.seed(SEED)
    Q = generate_sparse_householder(N, SPARSITY)
    D = block_diagonal(sample_ring(r_min, r_max, N // 2), N)
    return Q @ D @ Q.T


matrices = [build(r_min, r_max) for _, r_min, r_max in SPECTRA]
vmax = max(np.abs(W).max() for W in matrices)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
for ax, W, (name, _, _) in zip(axes, matrices, SPECTRA):
    im = ax.imshow(np.abs(W), cmap="viridis", vmin=0, vmax=vmax, interpolation="none")
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
plt.savefig(os.path.join(HERE, "heatmaps.png"), dpi=150)
plt.close()
