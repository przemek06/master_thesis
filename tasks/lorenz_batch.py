import numpy as np
from tasks.lorenz_refined import _generate_3d as _generate


def load(n_val, n_test, n_train=1, series_length=6000, subsample=1):
    def pairs(x):
        return x[:-1], x[1:]

    if n_train == 1:
        u_tr, y_tr = pairs(_generate(series_length, seed=0, label="train", subsample=subsample))
    else:
        train_list = [pairs(_generate(series_length, seed=i, label=f"train_{i}", subsample=subsample)) for i in range(n_train)]
        u_tr = np.stack([u for u, _ in train_list])
        y_tr = np.stack([y for _, y in train_list])

    val_list  = [pairs(_generate(series_length, seed=n_train + i,         label=f"val_{i}",  subsample=subsample)) for i in range(n_val)]
    test_list = [pairs(_generate(series_length, seed=n_train + n_val + i, label=f"test_{i}", subsample=subsample)) for i in range(n_test)]

    return u_tr, y_tr, val_list, test_list
