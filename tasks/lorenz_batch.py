import numpy as np
from tasks.lorenz_refined import _generate_3d as _generate


def load(n_val, n_test, n_train=1, series_length=6000, subsample=1, seed_offset=0):
    train_raw = [_generate(series_length, seed=seed_offset + i, subsample=subsample) for i in range(n_train)]
    stacked = np.concatenate(train_raw)
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)

    def pairs(x):
        x = (x - mean) / std
        return x[:-1], x[1:]

    if n_train == 1:
        u_tr, y_tr = pairs(train_raw[0])
    else:
        train_list = [pairs(x) for x in train_raw]
        u_tr = np.stack([u for u, _ in train_list])
        y_tr = np.stack([y for _, y in train_list])

    val_list = [pairs(_generate(series_length, seed=seed_offset + n_train + i, subsample=subsample)) for i in range(n_val)]
    test_list = [pairs(_generate(series_length, seed=seed_offset + n_train + n_val + i, subsample=subsample)) for i in range(n_test)]

    return u_tr, y_tr, val_list, test_list
