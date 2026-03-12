import numpy as np
import os

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "lorenz.npz")
SEED = 42
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.01
TOTAL = 120000
TRANSIENT = 10000


def _lorenz(state):
    x, y, z = state
    return np.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ])


def _generate():
    rng = np.random.default_rng(SEED)
    state = rng.standard_normal(3)
    trajectory = np.empty((TOTAL, 3))

    for t in range(TOTAL):
        k1 = _lorenz(state)
        k2 = _lorenz(state + 0.5 * DT * k1)
        k3 = _lorenz(state + 0.5 * DT * k2)
        k4 = _lorenz(state + DT * k3)
        state = state + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory[t] = state

    x = trajectory[TRANSIENT:, 0]
    x = (x - x.mean()) / x.std()
    return x


def load():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        x = _generate()
        np.savez(CACHE_PATH, x=x)
    else:
        x = np.load(CACHE_PATH)["x"]

    u = x[:-1].reshape(-1, 1)
    y = x[1:].reshape(-1, 1)

    n = len(u)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    return (
        u[:train_end], y[:train_end],
        u[train_end:val_end], y[train_end:val_end],
        u[val_end:], y[val_end:],
    )
