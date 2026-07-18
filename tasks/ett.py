import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ETDataset-main", "ETDataset-main", "ETT-small")
LOAD_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET_COL = "OT"
FILES = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]


def _steps_per_month(file):
    return 30 * 24 if file.startswith("ETTh") else 30 * 24 * 4


def _raw(file):
    import pandas as pd

    df = pd.read_csv(os.path.join(DATA_DIR, f"{file}.csv"))
    loads = df[LOAD_COLS].to_numpy(dtype=np.float32)
    ot = df[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
    return loads, ot


def load(file="ETTh1", horizon=0):
    loads, ot = _raw(file)
    m = _steps_per_month(file)
    train_end, val_end, test_end = 12 * m, 16 * m, 20 * m

    x = (loads - loads[:train_end].mean(0)) / loads[:train_end].std(0)
    y = (ot - ot[:train_end].mean(0)) / ot[:train_end].std(0)

    def seg(a, b):
        u, t = x[a:b], y[a:b]
        return (u[:len(u) - horizon], t[horizon:]) if horizon > 0 else (u, t)

    u_tr, y_tr = seg(0, train_end)
    val = seg(train_end, val_end)
    test = seg(val_end, test_end)
    return u_tr, y_tr, [val], [test]
