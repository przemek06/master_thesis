import numpy as np
import os

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "mackey_glass_refined.npz")
TAU = 17
BETA = 0.2
GAMMA = 0.1
N = 10
DELTA = 0.1
SUBSAMPLE = 10
FINE_TAU = round(TAU / DELTA)
TRANSIENT_FINE = 0


def _generate(coarse_length, seed, label=""):
    fine_length = coarse_length * SUBSAMPLE
    total = FINE_TAU + TRANSIENT_FINE + fine_length
    print(f"  Generating {label} ({fine_length} fine steps)...", flush=True)

    rng = np.random.default_rng(seed)
    x = np.zeros(total)
    x[:FINE_TAU] = 0.9 + 0.1 * rng.random(FINE_TAU)

    for t in range(FINE_TAU, total - 1):
        x_d = x[t - FINE_TAU]
        x[t + 1] = x[t] + DELTA * (BETA * x_d / (1 + x_d ** N) - GAMMA * x[t])

    x = x[FINE_TAU + TRANSIENT_FINE:]
    x = x[::SUBSAMPLE]
    print(f"  Done {label}.", flush=True)
    return np.tanh(x - 1)


def load():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        print("Generating Mackey-Glass refined sequences...", flush=True)
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
