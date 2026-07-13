import numpy as np
import os

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "lorenz_refined.npz")
SIGMA = 10.0
RHO = 28.0
BETA = 8.0 / 3.0
DT = 0.01
TRANSIENT = 5000


def _lorenz(state):
    x, y, z = state
    return np.array([
        SIGMA * (y - x),
        x * (RHO - z) - y,
        x * y - BETA * z,
    ])


def _generate(length, seed, label=""):
    print(f"  Generating {label} ({length} steps)...", flush=True)
    rng = np.random.default_rng(seed)
    state = rng.standard_normal(3)
    total = TRANSIENT + length
    trajectory = np.empty((total, 3))

    for t in range(total):
        k1 = _lorenz(state)
        k2 = _lorenz(state + 0.5 * DT * k1)
        k3 = _lorenz(state + 0.5 * DT * k2)
        k4 = _lorenz(state + DT * k3)
        state = state + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory[t] = state

    x = trajectory[TRANSIENT:, 0]
    x = (x - x.mean()) / x.std()
    print(f"  Done {label}.", flush=True)
    return x


def _generate_3d(length, seed, label="", subsample=1):
    print(f"  Generating {label} ({length} steps)...", flush=True)
    rng = np.random.default_rng(seed)
    state = rng.standard_normal(3)
    total = TRANSIENT + length * subsample
    trajectory = np.empty((total, 3))

    for t in range(total):
        k1 = _lorenz(state)
        k2 = _lorenz(state + 0.5 * DT * k1)
        k3 = _lorenz(state + 0.5 * DT * k2)
        k4 = _lorenz(state + DT * k3)
        state = state + (DT / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory[t] = state

    traj = trajectory[TRANSIENT::subsample][:length]
    traj = (traj - traj.mean(axis=0)) / traj.std(axis=0)
    print(f"  Done {label}.", flush=True)
    return traj


def load():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        print("Generating Lorenz refined sequences...", flush=True)
        data = {
            "train": _generate(3001, seed=0, label="train"),
            "val":   _generate(4001, seed=1, label="val"),
            "test":  _generate(4001, seed=2, label="test"),
        }
        np.savez(CACHE_PATH, **data)
        print("Saved to cache.", flush=True)
    else:
        data = np.load(CACHE_PATH)

    def pairs(x):
        return x[:-1].reshape(-1, 1), x[1:].reshape(-1, 1)

    u_tr, y_tr = pairs(data["train"])
    u_va, y_va = pairs(data["val"])
    u_te, y_te = pairs(data["test"])

    return u_tr, y_tr, u_va, y_va, u_te, y_te
