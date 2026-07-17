import numpy as np
import os

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "santafe_laser.npz")


def _generate():
    from reservoirpy.datasets import santafe_laser

    x = santafe_laser().astype(float).reshape(-1)
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
