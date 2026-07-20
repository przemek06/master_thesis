import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "jena_climate_2009_2016.csv")
COV_COLS = ["p (mbar)", "Tpot (K)", "Tdew (degC)", "rh (%)", "VPmax (mbar)",
            "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)",
            "rho (g/m**3)", "wv (m/s)", "max. wv (m/s)", "wd (deg)"]
TARGET_COL = "T (degC)"
TRAIN_F, VAL_F = 0.7, 0.1


def _raw():
    import pandas as pd

    df = pd.read_csv(os.path.join(DATA_DIR, "jena_climate_2009_2016.csv"))
    for c in ["wv (m/s)", "max. wv (m/s)"]:
        df.loc[df[c] == -9999.0, c] = 0.0
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
    n = len(covs)
    train_end, val_end, test_end = int(TRAIN_F * n), int((TRAIN_F + VAL_F) * n), n

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
