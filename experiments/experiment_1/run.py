import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from tasks.narma10 import load
from models.esn import ESN

HERE = os.path.dirname(__file__)


def nmse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true)


u_train, y_train, u_val, y_val, u_test, y_test = load()

esn = ESN(n_inputs=1, n_reservoir=500, n_outputs=1, spectral_radius=0.9, sparsity=0.9, ridge=1e-6, seed=0)
esn.fit(u_train, y_train, warmup=100)

y_pred = esn.predict(u_test)

print(f"Test NMSE: {nmse(y_test, y_pred):.6f}")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
for i, ax in enumerate(axes):
    start = i * 100
    end = start + 200
    ax.plot(y_test[start:end], label="Target", color="steelblue")
    ax.plot(y_pred[start:end], label="Prediction", color="tomato", linestyle="--")
    ax.set_title(f"Timesteps {start}–{end}")
    ax.legend(loc="upper right")

fig.suptitle("ESN — NARMA-10 predictions vs target")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.show()
