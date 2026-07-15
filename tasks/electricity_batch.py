import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), "data", "electricityloaddiagrams20112014", "LD2011_2014.txt")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "electricity_batch.npz")
MAX_ZERO_FRACTION = 0.001
TRAIN_YEAR = 2012
VAL_YEAR = 2013
TEST_YEAR = 2014
SEED = 0


def _generate():
    import pandas as pd

    print("Parsing electricity load diagrams...", flush=True)
    df = pd.read_csv(RAW_PATH, sep=";", decimal=",", index_col=0, parse_dates=True)
    raw = df.to_numpy(dtype=np.float64)
    keep = (raw[0] != 0) & ((raw == 0).mean(0) < MAX_ZERO_FRACTION)
    hourly = df.loc[:, keep].resample("1h", closed="right", label="left").mean()
    print(f"Kept {hourly.shape[1]} clients, {hourly.shape[0]} hourly steps.", flush=True)
    return hourly.to_numpy(dtype=np.float32), hourly.index.year.to_numpy()


def _panel():
    if not os.path.exists(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        panel, years = _generate()
        np.savez(CACHE_PATH, panel=panel, years=years)
    else:
        data = np.load(CACHE_PATH)
        panel, years = data["panel"], data["years"]

    return panel, years


def load(n_val, n_test, n_train=10, horizon=24, seed_offset=0):
    panel, years = _panel()

    reference = panel[years == TRAIN_YEAR]
    z = (panel - reference.mean(0)) / reference.std(0)

    rng = np.random.default_rng(SEED + seed_offset)
    perm = rng.permutation(z.shape[1])
    train_clients = perm[:n_train]
    val_clients = perm[n_train:n_train + n_val]
    test_clients = perm[n_train + n_val:n_train + n_val + n_test]

    def pairs(client, year):
        s = z[years == year, client]
        return s[:len(s) - horizon].reshape(-1, 1), s[horizon:].reshape(-1, 1)

    if n_train == 1:
        u_tr, y_tr = pairs(train_clients[0], TRAIN_YEAR)
    else:
        train_pairs = [pairs(c, TRAIN_YEAR) for c in train_clients]
        u_tr = np.stack([u for u, _ in train_pairs])
        y_tr = np.stack([y for _, y in train_pairs])

    val_list = [pairs(c, VAL_YEAR) for c in val_clients]
    test_list = [pairs(c, TEST_YEAR) for c in test_clients]

    return u_tr, y_tr, val_list, test_list
