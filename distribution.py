import numpy as np


def sample_von_mises(mu: float, kappa: float, size: int) -> np.ndarray:
    return np.random.vonmises(mu, kappa, size)


def sample_skew_normal(a: float, loc: float, scale: float, size: int) -> np.ndarray:
    from scipy.stats import skewnorm
    return skewnorm.rvs(a, loc=loc, scale=scale, size=size)


def sample_eigenvalues(r: float, alpha: float, beta: float, mu: float, kappa: float, size: int, radial_fn=None, angular_fn=None) -> np.ndarray:
    radii = radial_fn(size) if radial_fn is not None else np.random.beta(alpha, beta, size) * r
    angles = angular_fn(size) if angular_fn is not None else sample_von_mises(mu, kappa, size)
    return radii * np.exp(1j * angles)


def sample_eigenvalues_real(r_min: float, r_max: float, alpha: float, size: int) -> np.ndarray:
    u = np.random.uniform(0, 1, size)
    r = r_min + (r_max - r_min) * u ** (1 / alpha)
    signs = np.random.choice([-1.0, 1.0], size=size)
    return r * signs


def sample_eigenvalues_bimodal(r_low: float, r_high: float, ratio: float, size: int) -> np.ndarray:
    half = size // 2
    n_low = int(half * ratio)
    n_high = half - n_low
    angles_low = np.random.uniform(0, 2 * np.pi, n_low)
    angles_high = np.random.uniform(0, 2 * np.pi, n_high)
    eigs = np.concatenate([r_low * np.exp(1j * angles_low), r_high * np.exp(1j * angles_high)])
    return np.concatenate([eigs, eigs.conj()])


def sample_eigenvalues_uniform_ring(r: float, size: int) -> np.ndarray:
    half = size // 2
    angles = np.random.uniform(0, 2 * np.pi, half)
    eigs = r * np.exp(1j * angles)
    return np.concatenate([eigs, eigs.conj()])


def sample_eigenvalues_ginibre(r_min: float, r_max: float, alpha: float, size: int) -> np.ndarray:
    half = size // 2
    G = (np.random.randn(half, half) + 1j * np.random.randn(half, half)) / np.sqrt(2 * half)
    eigs = np.linalg.eigvals(G)
    r = np.abs(eigs)
    theta = np.angle(eigs)
    r_hat = r_min + (r_max - r_min) * r ** (1 / alpha)
    transformed = r_hat * np.exp(1j * theta)
    return np.concatenate([transformed, transformed.conj()])
