import numpy as np

K_MAX = 200


def _generate(length, seed, label=""):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-0.8, 0.8, length + K_MAX)
    y = np.stack([u[K_MAX - k: K_MAX - k + length] for k in range(1, K_MAX + 1)], axis=1)
    return u[K_MAX:].reshape(-1, 1), y


def load(n_val, n_test, n_train=1, series_length=6000, seed_offset=0):
    if n_train == 1:
        u_tr, y_tr = _generate(series_length, seed=seed_offset, label="train")
    else:
        train_list = [_generate(series_length, seed=seed_offset + i, label=f"train_{i}") for i in range(n_train)]
        u_tr = np.stack([u for u, _ in train_list])
        y_tr = np.stack([y for _, y in train_list])

    val_list = [_generate(series_length, seed=seed_offset + n_train + i, label=f"val_{i}") for i in range(n_val)]
    test_list = [_generate(series_length, seed=seed_offset + n_train + n_val + i, label=f"test_{i}") for i in range(n_test)]

    return u_tr, y_tr, val_list, test_list
