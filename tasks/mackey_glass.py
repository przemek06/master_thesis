import numpy as np
import os

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "mackey_glass.npz")
SEED = 42
TAU = 17
BETA = 0.2
GAMMA = 0.1
N = 10
TOTAL = 15001
TRANSIENT = 3000


def _generate():
    rng = np.random.default_rng(SEED)
    x = np.zeros(TOTAL + TAU)
    x[:TAU] = 0.9 + 0.1 * rng.random(TAU)

    for t in range(TAU, TOTAL + TAU - 1):
        x_delayed = x[t - TAU]
        x[t + 1] = x[t] + BETA * x_delayed / (1 + x_delayed ** N) - GAMMA * x[t]

    return x[TRANSIENT + TAU:]


def load():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        x = _generate()
        np.savez(CACHE_PATH, x=x)
    else:
        x = np.load(CACHE_PATH)["x"]

    u = x[:-1].reshape(-1, 1)
    y = x[1:].reshape(-1, 1)

    return (
        u[:7200], y[:7200],
        u[7200:9600], y[7200:9600],
        u[9600:], y[9600:],
    )
