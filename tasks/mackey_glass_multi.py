import numpy as np
from tasks.mackey_glass_refined import _generate

SERIES_LENGTH = 6001


def load(n_val, n_test, n_train=1, seed_offset=0):
    def pairs(x):
        return x[:-1].reshape(-1, 1), x[1:].reshape(-1, 1)

    if n_train == 1:
        u_tr, y_tr = pairs(_generate(SERIES_LENGTH, seed=seed_offset))
    else:
        train_list = [pairs(_generate(SERIES_LENGTH, seed=seed_offset + i)) for i in range(n_train)]
        u_tr = np.stack([u for u, _ in train_list])
        y_tr = np.stack([y for _, y in train_list])

    val_list = [pairs(_generate(SERIES_LENGTH, seed=seed_offset + n_train + i)) for i in range(n_val)]
    test_list = [pairs(_generate(SERIES_LENGTH, seed=seed_offset + n_train + n_val + i)) for i in range(n_test)]

    return u_tr, y_tr, val_list, test_list
