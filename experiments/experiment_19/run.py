import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
from generate import generate_isospectral_sparse_orthogonal_matrix
from distribution import sample_eigenvalues_ginibre
from analyze import get_eigenvalues, get_sparsity
from visualize import plot_heatmap, plot_circle_with_points

HERE = os.path.dirname(__file__)

N = 200
R_MIN = 0.5
R_MAX = 0.95
ALPHA = 2.0
SPARSITY = 0.95
ITERATIONS = 100
THRESHOLD = 0.03
SEED = 42

eigenvalue_fn = lambda size: sample_eigenvalues_ginibre(R_MIN, R_MAX, ALPHA, size)

Q = generate_isospectral_sparse_orthogonal_matrix(
    eigenvalue_fn,
    N,
    SPARSITY,
    iterations=ITERATIONS,
    threshold=THRESHOLD,
    seed=SEED,
)

eigs = eigenvalue_fn(N)
D = np.diag(eigs)
W = Q.astype(complex) @ D @ Q.astype(complex).conj().T

print(f"W sparsity: {get_sparsity(W):.4f}")
print(f"Diagonal ratio: {np.sum(np.abs(np.diag(W))) / np.sum(np.abs(W)):.4f}")

plot_heatmap([np.abs(W)], os.path.join(HERE, "heatmap.png"))
plot_circle_with_points(R_MAX, get_eigenvalues(W), os.path.join(HERE, "eigenvalues.png"))
