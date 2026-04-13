import numpy as np
import os
from tasks.lorenz_refined import _generate

CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "lorenz_batch.npz")
SERIES_LENGTH = 6001
N_VAL = 10
N_TEST = 10


def load():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        data = {"train": _generate(SERIES_LENGTH, seed=0, label="train")}
        for i in range(N_VAL):
            data[f"val_{i}"] = _generate(SERIES_LENGTH, seed=1 + i, label=f"val_{i}")
        for i in range(N_TEST):
            data[f"test_{i}"] = _generate(SERIES_LENGTH, seed=1 + N_VAL + i, label=f"test_{i}")
        np.savez(CACHE_PATH, **data)
    else:
        data = np.load(CACHE_PATH)

    def pairs(x):
        return x[:-1].reshape(-1, 1), x[1:].reshape(-1, 1)

    u_tr, y_tr = pairs(data["train"])
    val_list = [pairs(data[f"val_{i}"]) for i in range(N_VAL)]
    test_list = [pairs(data[f"test_{i}"]) for i in range(N_TEST)]

    return u_tr, y_tr, val_list, test_list
