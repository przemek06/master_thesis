import numpy as np
import os

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "narma10.npz")
SEED = 42
N = 10
TOTAL = 10000


def _generate():
    rng = np.random.default_rng(SEED)
    u = rng.uniform(0, 0.5, TOTAL + N)
    y = np.zeros(TOTAL + N)

    for t in range(N, TOTAL + N - 1):
        y[t + 1] = (
            0.3 * y[t]
            + 0.05 * y[t] * np.sum(y[t - N + 1:t + 1])
            + 1.5 * u[t - N + 1] * u[t]
            + 0.1
        )

    return u[N:], y[N:]


def load():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        u, y = _generate()
        np.savez(CACHE_PATH, u=u, y=y)
    else:
        data = np.load(CACHE_PATH)
        u, y = data["u"], data["y"]

    u = u.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return (
        u[:6000], y[:6000],
        u[6000:8000], y[6000:8000],
        u[8000:], y[8000:],
    )
