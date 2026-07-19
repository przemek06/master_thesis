import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
COV_COLS = ["Visibility", "DryBulbFarenheit", "DryBulbCelsius", "DewPointFarenheit",
            "DewPointCelsius", "RelativeHumidity", "WindSpeed", "WindDirection",
            "StationPressure", "Altimeter"]
TARGET_COL = "WetBulbCelsius"
STEPS_PER_MONTH = 30 * 24
TRAIN_M, VAL_M, TEST_M = 28, 10, 10


def _raw():
    import pandas as pd

    df = pd.read_csv(os.path.join(DATA_DIR, "WTH.csv"))
    covs = df[COV_COLS].to_numpy(dtype=np.float32)
    ot = df[TARGET_COL].to_numpy(dtype=np.float32).reshape(-1, 1)
    return covs, ot


def _lag_window(a, window):
    n = len(a)
    return np.concatenate(
        [np.concatenate([np.zeros((k, a.shape[1]), a.dtype), a[:n - k]]) for k in range(window)],
        axis=1,
    )


def load(horizon=24, lag=24):
    covs, ot = _raw()
    m = STEPS_PER_MONTH
    train_end, val_end, test_end = TRAIN_M * m, (TRAIN_M + VAL_M) * m, (TRAIN_M + VAL_M + TEST_M) * m

    covs_std = (covs - covs[:train_end].mean(0)) / covs[:train_end].std(0)
    y = (ot - ot[:train_end].mean(0)) / ot[:train_end].std(0)
    x = np.concatenate([covs_std, _lag_window(y, lag)], axis=1)

    def seg(a, b):
        u, t = x[a:b], y[a:b]
        return (u[:len(u) - horizon], t[horizon:]) if horizon > 0 else (u, t)

    u_tr, y_tr = seg(0, train_end)
    val = seg(train_end, val_end)
    test = seg(val_end, test_end)
    return u_tr, y_tr, [val], [test]
