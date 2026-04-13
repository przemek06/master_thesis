import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import schur
from scipy.optimize import linear_sum_assignment
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)

T_template = None
Z_template = None


def precompute(sparsity_mask, n_iter=100, seed=42):
    global T_template, Z_template

    n = sparsity_mask.shape[0]
    rng = np.random.default_rng(seed)

    radii = rng.uniform(0.1, 0.99, n)
    angles = rng.uniform(0, 2 * np.pi, n)
    template_eigs = radii * np.exp(1j * angles)

    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    W = (Q @ np.diag(template_eigs) @ Q.conj().T) * sparsity_mask

    for _ in range(n_iter):
        T, Z = schur(W, output='complex')
        actual_diag = np.diag(T)
        cost = np.abs(actual_diag[:, None] - template_eigs[None, :])
        row_ind, col_ind = linear_sum_assignment(cost)
        new_diag = np.empty(n, dtype=complex)
        new_diag[row_ind] = template_eigs[col_ind]
        np.fill_diagonal(T, new_diag)
        W = (Z @ T @ Z.conj().T) * sparsity_mask

    T_template, Z_template = schur(W, output='complex')


def generate(sparsity_mask, eigenvalues, n_refine=100):
    n = len(eigenvalues)
    T = T_template.copy()
    cost = np.abs(np.diag(T)[:, None] - eigenvalues[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    new_diag = np.empty(n, dtype=complex)
    new_diag[row_ind] = eigenvalues[col_ind]
    np.fill_diagonal(T, new_diag)
    W = (Z_template @ T @ Z_template.conj().T) * sparsity_mask

    for _ in range(n_refine):
        T, Z = schur(W, output='complex')
        actual_diag = np.diag(T)
        cost = np.abs(actual_diag[:, None] - eigenvalues[None, :])
        row_ind, col_ind = linear_sum_assignment(cost)
        new_diag = np.empty(n, dtype=complex)
        new_diag[row_ind] = eigenvalues[col_ind]
        np.fill_diagonal(T, new_diag)
        W = (Z @ T @ Z.conj().T) * sparsity_mask

    return W


def test(matrix, sparsity_mask, eigenvalues):
    diag_ratio = np.sum(np.abs(np.diag(matrix))) / np.sum(np.abs(matrix))
    sparsity = np.mean(matrix == 0)

    actual_eigs = np.linalg.eigvals(matrix)
    cost = np.abs(actual_eigs[:, None] - eigenvalues[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    eig_diff = np.mean(cost[row_ind, col_ind])

    actual_zero = matrix == 0
    given_zero = sparsity_mask == 0
    mask_match = np.mean(actual_zero == given_zero)

    print(f"Diagonal ratio:  {diag_ratio:.4f}")
    print(f"Sparsity:        {sparsity:.4f}")
    print(f"Eigenvalue diff: {eig_diff:.6f}")
    print(f"Mask match:      {mask_match:.4f}")


N = 500
SPARSITY = 0.98
R_MIN = 0.5
R_MAX = 0.99
ALPHA = 2.0
rng = np.random.default_rng(42)

sparsity_mask = rng.random((N, N)) > SPARSITY
eigenvalues = sample_eigenvalues_ginibre(R_MIN, R_MAX, ALPHA, size=N)

precompute(sparsity_mask)

start = time.perf_counter()
matrix = generate(sparsity_mask, eigenvalues)
elapsed = time.perf_counter() - start
print(f"generate() time: {elapsed:.6f}s")
test(matrix, sparsity_mask, eigenvalues)

actual_eigs = np.linalg.eigvals(matrix)
r = R_MAX

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, eigs, title, color in zip(axes[:2], [eigenvalues, actual_eigs], ["Given", "Actual"], ["steelblue", "tomato"]):
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), r, fill=False, color="gray", linestyle="--"))
    ax.scatter(eigs.real, eigs.imag, s=10, color=color)
    ax.set_title(title)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

axes[2].set_aspect("equal")
axes[2].add_patch(plt.Circle((0, 0), r, fill=False, color="gray", linestyle="--"))
axes[2].scatter(eigenvalues.real, eigenvalues.imag, s=10, color="steelblue", label="Given", alpha=0.7)
axes[2].scatter(actual_eigs.real, actual_eigs.imag, s=10, color="tomato", label="Actual", alpha=0.7)
axes[2].set_title("Overlaid")
axes[2].set_xlim(-1.5, 1.5)
axes[2].set_ylim(-1.5, 1.5)
axes[2].legend()

fig.suptitle("Eigenvalues: given vs actual after sparse isospectral generation")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, mat, title in zip(axes, [np.diag(eigenvalues), matrix], ["Given (diagonal)", "Generated"]):
    im = ax.imshow(np.abs(mat), cmap="hot")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)

fig.suptitle("Heatmap: given vs generated sparse matrix")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "heatmap.png"), dpi=150)
plt.close(fig)
