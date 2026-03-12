import numpy as np


def get_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.eigvals(matrix)


def get_sparsity(matrix: np.ndarray) -> float:
    m = np.asarray(matrix)
    return np.sum(m == 0) / m.size
