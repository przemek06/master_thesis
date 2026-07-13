import numpy as np
from tasks.mackey_glass_refined import _generate

SERIES_LENGTH = 6001


def load(n_val, n_test, n_train=1):
    def pairs(x):
        return x[:-1].reshape(-1, 1), x[1:].reshape(-1, 1)

    if n_train == 1:
        u_tr, y_tr = pairs(_generate(SERIES_LENGTH, seed=0, label="train"))
    else:
        train_list = [pairs(_generate(SERIES_LENGTH, seed=i, label=f"train_{i}")) for i in range(n_train)]
        u_tr = np.stack([u for u, _ in train_list])
        y_tr = np.stack([y for _, y in train_list])

    val_list = [pairs(_generate(SERIES_LENGTH, seed=n_train + i, label=f"val_{i}")) for i in range(n_val)]
    test_list = [pairs(_generate(SERIES_LENGTH, seed=n_train + n_val + i, label=f"test_{i}")) for i in range(n_test)]

    return u_tr, y_tr, val_list, test_list
