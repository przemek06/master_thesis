import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import torch
import matplotlib.pyplot as plt
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient
from analyze import get_eigenvalues, get_sparsity
from generate import generate_matrix_from_eigenvalues
from distribution import sample_eigenvalues

HERE = os.path.dirname(__file__)

eta = 0.05
N = 100
MASK_SPARSITY = 0.98
THRESHOLD_1 = 0.01
THRESHOLD_2 = 0.5
ITERATIONS = 10000
SEED = 42

rng = np.random.default_rng(SEED)
W = generate_matrix_from_eigenvalues(lambda size: sample_eigenvalues(1.05, alpha=0.1, beta=0.1, mu=0.0, kappa=0.0, size=size), N, sparsity=0.0)

mask = (rng.random((N, N)) > MASK_SPARSITY).astype(np.float64)

W_t = torch.tensor(W, dtype=torch.float64)
mask_t = torch.tensor(1 - mask, dtype=torch.float64)

manifold = Stiefel(N, N)

@pymanopt.function.pytorch(manifold)
def cost(Q):
    return torch.sum((Q @ W_t @ Q.T * mask_t) ** 2)

problem = pymanopt.Problem(manifold, cost)
optimizer = ConjugateGradient(max_iterations=ITERATIONS, verbosity=2)
result = optimizer.run(problem)

Q = result.pointz   
print(np.max(np.abs(Q)))
Q[np.abs(Q) < THRESHOLD_1] = 0
W_after = Q @ W @ Q.T
W_after[np.abs(W_after) < THRESHOLD_2] = 0

print(f"Sparsity before: {get_sparsity(W):.4f}")
print(f"Sparsity after:  {get_sparsity(W_after):.4f}")

eigenvalues_before = get_eigenvalues(W)
eigenvalues_after = get_eigenvalues(W_after)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
r = 1.0
for ax, eigs, title, color in zip(axes[:2], [eigenvalues_before, eigenvalues_after], ["Before", "After"], ["steelblue", "tomato"]):
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), r, fill=False, color="gray", linestyle="--"))
    ax.scatter(eigs.real, eigs.imag, s=10, color=color)
    ax.set_title(title)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

axes[2].set_aspect("equal")
axes[2].add_patch(plt.Circle((0, 0), r, fill=False, color="gray", linestyle="--"))
axes[2].scatter(eigenvalues_before.real, eigenvalues_before.imag, s=10, color="steelblue", label="Before", alpha=0.7)
axes[2].scatter(eigenvalues_after.real, eigenvalues_after.imag, s=10, color="tomato", label="After", alpha=0.7)
axes[2].set_title("Overlaid")
axes[2].set_xlim(-1.5, 1.5)
axes[2].set_ylim(-1.5, 1.5)
axes[2].legend()

fig.suptitle("Eigenvalues before and after isospectral sparsification")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, mat, title in zip(axes, [W, W_after], ["Before", "After"]):
    im = ax.imshow(np.abs(mat), cmap="hot")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

fig.suptitle("Heatmap before and after isospectral sparsification")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "heatmap.png"), dpi=150)
plt.show()
