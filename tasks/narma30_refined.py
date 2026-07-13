import numpy as np
import os

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "narma30_refined.npz")
N = 30


def _generate(length, seed, label=""):
    print(f"  Generating {label} ({length} steps)...", flush=True)
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 0.5, length + N)
    y = np.zeros(length + N)

    for t in range(N, length + N - 1):
        y[t + 1] = (
            0.2 * y[t]
            + 0.04 * y[t] * np.sum(y[t - N + 1:t + 1])
            + 1.5 * u[t - N + 1] * u[t]
            + 0.001
        )

    print(f"  Done {label}.", flush=True)
    return u[N:].reshape(-1, 1), y[N:].reshape(-1, 1)


def load():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        print("Generating NARMA-30 refined sequences...", flush=True)
        u_tr, y_tr = _generate(6000, seed=0, label="train")
        u_va, y_va = _generate(2000, seed=1, label="val")
        u_te, y_te = _generate(2000, seed=2, label="test")
        np.savez(CACHE_PATH, u_tr=u_tr, y_tr=y_tr, u_va=u_va, y_va=y_va, u_te=u_te, y_te=y_te)
        print("Saved to cache.", flush=True)
    else:
        data = np.load(CACHE_PATH)
        u_tr, y_tr = data["u_tr"], data["y_tr"]
        u_va, y_va = data["u_va"], data["y_va"]
        u_te, y_te = data["u_te"], data["y_te"]

    return u_tr, y_tr, u_va, y_va, u_te, y_te
