import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)

N = 200
SPARSITY = 0.9
ALPHA = 1.0
SEED = 0
ITERATIONS = 50
THRESHOLD = 0.01

SPECTRA = [
    ("big ring", 0.85, 0.90),
    ("small ring", 0.25, 0.30),
]


def build(r_min, r_max):
    np.random.seed(SEED)
    return generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=r_min, r_max=r_max, alpha=ALPHA, size=size),
        N, SPARSITY, iterations=ITERATIONS, threshold=THRESHOLD, seed=SEED,
    )


matrices = [np.abs(build(r_min, r_max)) for _, r_min, r_max in SPECTRA]
vmax = max(W.max() for W in matrices)

fig, axes = plt.subplots(1, len(SPECTRA), figsize=(13, 6.2))
for ax, W, (name, _, _) in zip(axes, matrices, SPECTRA):
    im = ax.imshow(W, cmap="viridis", vmin=0, vmax=vmax, interpolation="none")
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
plt.savefig(os.path.join(HERE, "iesn_heatmaps.png"), dpi=150)
plt.close()
