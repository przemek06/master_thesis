from generate import generate_sparse_matrix, generate_sparse_householder, generate_taylor_exp_matrix, generate_composite_matrix, generate_block_diagonal_rotations, generate_block_diagonal_fourier, generate_matrix_from_eigenvalues, generate_analytic_map_matrix
from distribution import sample_eigenvalues
from analyze import get_eigenvalues, get_sparsity
from visualize import plot_circle_with_points, plot_heatmap
import numpy as np
import os

def main():
    os.makedirs("images", exist_ok=True)
    n = 500
    sparsity = 0.9
    r = 1.0
    eta = 0.05

    uniform_matrix = generate_sparse_matrix(n, sparsity, lambda k: np.random.uniform(-0.5, 0.5, k))
    normal_matrix = generate_sparse_matrix(n, sparsity, lambda k: np.random.normal(0, 1, k))
    householder_matrix = generate_sparse_householder(n, sparsity)
    taylor_matrix = generate_taylor_exp_matrix(n, m=5, sparsity=0.999)
    composite_matrix = generate_composite_matrix(n, sparsity, m=20)
    rotation_matrix = generate_block_diagonal_rotations(n)
    fourier_matrix = generate_block_diagonal_fourier(n, block_size=10)
    prescribed_matrix = generate_matrix_from_eigenvalues(lambda size: sample_eigenvalues(r+eta, alpha=0.1, beta=0.1, mu=0.0, kappa=0.0, size=size), n, sparsity)
    analytic_map_matrix = generate_analytic_map_matrix(n, sparsity)

    for matrix, name in [
        (uniform_matrix.toarray(), "uniform"),
        (normal_matrix.toarray(), "normal"),
        (householder_matrix, "householder"),
        (taylor_matrix, "taylor"),
        (composite_matrix, "composite"),
        (rotation_matrix, "rotation"),
        (fourier_matrix, "fourier"),
        (prescribed_matrix, "prescribed"),
        (analytic_map_matrix, "analytic_map"),
    ]:
        print(f"{name}: sparsity={get_sparsity(matrix):.4f}")

    eigenvalues_uniform = get_eigenvalues(uniform_matrix.toarray())
    eigenvalues_normal = get_eigenvalues(normal_matrix.toarray())
    eigenvalues_householder = get_eigenvalues(householder_matrix)
    eigenvalues_taylor = get_eigenvalues(taylor_matrix)
    eigenvalues_composite = get_eigenvalues(composite_matrix)
    eigenvalues_rotation = get_eigenvalues(rotation_matrix)
    eigenvalues_fourier = get_eigenvalues(fourier_matrix)
    eigenvalues_prescribed = get_eigenvalues(prescribed_matrix)
    eigenvalues_analytic_map = get_eigenvalues(analytic_map_matrix)

    for eigenvalues, name in [
        (eigenvalues_uniform, "uniform"),
        (eigenvalues_normal, "normal"),
        (eigenvalues_householder, "householder"),
        (eigenvalues_taylor, "taylor"),
        (eigenvalues_composite, "composite"),
        (eigenvalues_rotation, "rotation"),
        (eigenvalues_fourier, "fourier"),
        (eigenvalues_analytic_map, "analytic_map"),
    ]:
        eigenvalues *= r / np.max(np.abs(eigenvalues))
        plot_circle_with_points(r, eigenvalues, f"images/eigenvalues_{name}.png")

    plot_circle_with_points(r, eigenvalues_prescribed, "images/eigenvalues_prescribed.png")

    for gen_fn, name in [
        (lambda: generate_sparse_matrix(n, sparsity, lambda k: np.random.uniform(-0.5, 0.5, k)).toarray(), "uniform"),
        (lambda: generate_sparse_matrix(n, sparsity, lambda k: np.random.normal(0, 1, k)).toarray(), "normal"),
        (lambda: generate_sparse_householder(n, sparsity), "householder"),
        (lambda: generate_taylor_exp_matrix(n, m=5, sparsity=sparsity), "taylor"),
        (lambda: generate_composite_matrix(n, sparsity, m=20), "composite"),
        (lambda: generate_block_diagonal_rotations(n), "rotation"),
        (lambda: generate_block_diagonal_fourier(n, block_size=10), "fourier"),
        (lambda: generate_matrix_from_eigenvalues(lambda size: sample_eigenvalues(r+eta, alpha=15, beta=1.0, mu=0.0, kappa=0.0, size=size), n, sparsity), "prescribed"),
        (lambda: generate_analytic_map_matrix(n), "analytic_map"),
    ]:
        plot_heatmap([gen_fn()], f"images/heatmap_{name}.png")

if __name__ == "__main__":
    main()
