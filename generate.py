import numpy as np
import scipy.sparse as sp
from distribution import sample_eigenvalues


def generate_sparse_matrix(n, sparsity, dist_fn):
    mask = np.random.rand(n, n) > sparsity
    row, col = np.where(mask)
    data = dist_fn(len(row))
    return sp.csr_matrix((data, (row, col)), shape=(n, n))


def generate_sparse_householder(n, sparsity):
    Q = np.eye(n)
    k = max(1, int(n * (1 - sparsity)))
    for _ in range(k):
        v = np.random.randn(n)
        v /= np.linalg.norm(v)
        Q -= 2 * np.outer(v, v) @ Q
    threshold = np.percentile(np.abs(Q), sparsity * 100)
    Q[np.abs(Q) < threshold] = 0
    return Q


def generate_taylor_exp_matrix(n, m=5, sparsity=0.999):
    mask = np.random.rand(n, n) > sparsity
    A = np.random.randn(n, n) * mask
    norm = np.linalg.norm(A, 2)
    if norm > 0:
        A /= norm
    W = np.eye(n)
    Ak = np.eye(n)
    factorial = 1
    for k in range(1, m + 1):
        Ak = Ak @ A
        factorial *= k
        W += Ak / factorial
    return W


def generate_composite_matrix(n, sparsity, m=20):
    W = np.eye(n)
    for _ in range(m):
        mask = np.random.rand(n, n) > sparsity
        A = np.random.randn(n, n) * mask
        norm = np.linalg.norm(A, 2)
        scale = norm if norm > 0 else 1.0
        W = W @ (np.eye(n) + A / scale / m)
    return W


def generate_block_diagonal_rotations(n):
    W = np.zeros((n, n))
    i = 0
    while i + 1 < n:
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        W[i:i+2, i:i+2] = [[c, -s], [s, c]]
        i += 2
    if i < n:
        W[i, i] = np.random.choice([-1.0, 1.0])
    return W


def generate_block_diagonal_fourier(n, block_size=10):
    W = np.zeros((n, n), dtype=complex)
    i = 0
    while i < n:
        size = min(block_size, n - i)
        F = np.fft.fft(np.eye(size), axis=0) / np.sqrt(size)
        W[i:i+size, i:i+size] = F
        i += size
    return W


def generate_matrix_from_eigenvalues(eigenvalue_fn, n, sparsity):
    half = n // 2
    eigs = eigenvalue_fn(half)
    D = np.zeros((n, n))
    for i, lam in enumerate(eigs):
        a, b = lam.real, lam.imag
        D[2*i:2*i+2, 2*i:2*i+2] = [[a, -b], [b, a]]
    if n % 2 == 1:
        D[-1, -1] = eigenvalue_fn(1)[0].real
    Q, _ = np.linalg.qr(np.random.randn(n, n))
    W = Q @ D @ Q.T
    if sparsity > 0:
        threshold = np.percentile(np.abs(W), sparsity * 100)
        W[np.abs(W) < threshold] = 0
    return W


def _stiefel_optimize(Lambda, n, sparsity, iterations, seed):
    import torch
    import pymanopt
    from pymanopt.manifolds import Stiefel
    from pymanopt.optimizers import ConjugateGradient

    rng = np.random.default_rng(seed)
    mask = (rng.random((n, n)) > sparsity).astype(np.float64)
    mask_t = torch.tensor(1 - mask, dtype=torch.float64)
    Lambda_t = torch.tensor(Lambda, dtype=torch.complex128)

    manifold = Stiefel(n, n)

    @pymanopt.function.pytorch(manifold)
    def cost(Q):
        Q_c = Q.to(torch.complex128)
        X = Q_c @ Lambda_t @ Q_c.conj().T
        return torch.sum((torch.abs(X) * mask_t) ** 2)

    problem = pymanopt.Problem(manifold, cost)
    optimizer = ConjugateGradient(max_iterations=iterations, verbosity=0)
    result = optimizer.run(problem)
    return result.point


def generate_isospectral_sparse_matrix(eigenvalue_fn, n, sparsity, iterations=200, threshold=0.01, seed=None):
    eigs = eigenvalue_fn(n)
    Lambda = np.diag(eigs)
    Q = _stiefel_optimize(Lambda, n, sparsity, iterations, seed)
    W = Q.astype(complex) @ Lambda @ Q.astype(complex).conj().T
    W[np.abs(W) < threshold] = 0
    return W


def generate_isospectral_sparse_orthogonal_matrix(eigenvalue_fn, n, sparsity, iterations=200, threshold=0.01, seed=None, n_samples=10, diag_penalty=20.0):
    import torch
    import pymanopt
    from pymanopt.manifolds import Stiefel
    from pymanopt.optimizers import ConjugateGradient

    rng = np.random.default_rng(seed)
    mask = (rng.random((n, n)) > sparsity).astype(np.float64)
    mask_t = torch.tensor(1 - mask, dtype=torch.float64)
    Lambdas_t = [torch.tensor(np.diag(eigenvalue_fn(n)), dtype=torch.complex128) for _ in range(n_samples)]

    manifold = Stiefel(n, n)

    @pymanopt.function.pytorch(manifold)
    def cost(Q):
        Q_c = Q.to(torch.complex128)
        total = torch.tensor(0.0, dtype=torch.float64)
        for Lambda_t in Lambdas_t:
            X = Q_c @ Lambda_t @ Q_c.conj().T
            total = total + torch.sum((torch.abs(X) * mask_t) ** 2)
            total = total + diag_penalty * torch.sum(torch.abs(torch.diag(X)) ** 2)
        return total

    problem = pymanopt.Problem(manifold, cost)
    optimizer = ConjugateGradient(max_iterations=iterations, verbosity=0)
    result = optimizer.run(problem)
    Q = result.point
    Q[np.abs(Q) < threshold] = 0
    return Q


def generate_analytic_map_matrix(n, sparsity=0.9):
    first_row = np.array([np.cos(k) * np.exp(-0.1 * k) for k in range(n)])
    first_row /= np.max(np.abs(first_row)) + 1e-8
    W = np.array([np.roll(first_row, i) for i in range(n)])
    threshold = np.percentile(np.abs(W), sparsity * 100)
    W[np.abs(W) < threshold] = 0
    return W
