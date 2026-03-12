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
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)

N = 200
R_MIN = 0.5
R_MAX = 0.95
ALPHA = 2.0
MASK_SPARSITY = 0.9
THRESHOLD = 0.01
ITERATIONS = 200
SEED = 42

np.random.seed(SEED)
rng = np.random.default_rng(SEED)
eigenvalues = sample_eigenvalues_ginibre(R_MIN, R_MAX, ALPHA, size=N)

Lambda = np.diag(eigenvalues)
Lambda_t = torch.tensor(Lambda, dtype=torch.complex128)

mask = (rng.random((N, N)) > MASK_SPARSITY).astype(np.float64)
mask_t = torch.tensor(1 - mask, dtype=torch.float64)

manifold = Stiefel(N, N)

@pymanopt.function.pytorch(manifold)
def cost(Q):
    Q_c = Q.to(torch.complex128)
    X = Q_c @ Lambda_t @ Q_c.conj().T
    return torch.sum((torch.abs(X) * mask_t) ** 2)

problem = pymanopt.Problem(manifold, cost)
optimizer = ConjugateGradient(max_iterations=ITERATIONS, verbosity=2)
result = optimizer.run(problem)

Q = result.point
X_after = Q.astype(complex) @ Lambda @ Q.astype(complex).conj().T
X_after[np.abs(X_after) < THRESHOLD] = 0

should_be_zero = mask == 0
correctly_masked = np.sum((np.abs(X_after) == 0) & should_be_zero) / np.sum(should_be_zero)

print(f"Sparsity before (diagonal): {get_sparsity(Lambda):.4f}")
print(f"Sparsity after:             {get_sparsity(X_after):.4f}")
print(f"Correctly masked entries:   {correctly_masked:.2%}")

eigenvalues_before = get_eigenvalues(Lambda)
eigenvalues_after = get_eigenvalues(X_after)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
r = 1.0
for ax, eigs, title, color in zip(axes[:2], [eigenvalues_before, eigenvalues_after], ["Before (Λ)", "After"], ["steelblue", "tomato"]):
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

fig.suptitle("Eigenvalues before and after masked isospectral sparsification (Ginibre)")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, mat, title in zip(axes, [Lambda, X_after], ["Before (Λ)", "After"]):
    im = ax.imshow(np.abs(mat), cmap="hot")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

fig.suptitle("Heatmap before and after masked isospectral sparsification (Ginibre)")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "heatmap.png"), dpi=150)
plt.show()
