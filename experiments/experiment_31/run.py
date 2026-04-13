import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from tasks.mackey_glass_refined import load
from models.esn_customizable import ESNCustomizable

HERE = os.path.dirname(__file__)
WARMUP = 1000

def denormalize(x):
    return np.arctanh(np.clip(x, -1 + 1e-7, 1 - 1e-7)) + 1

def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)

u_train, y_train, u_val, y_val, u_test, y_test = load()

N_RESERVOIR = 1000
N_INPUTS = 1
N_OUTPUTS = 1
SEED = 0

rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))

model = ESNCustomizable(
    n_inputs=N_INPUTS,
    n_reservoir=N_RESERVOIR,
    n_outputs=N_OUTPUTS,
    spectral_radius=0.9993760476875644,
    sparsity=0.9697222820893018,
    leaky_rate=0.7987598599205421,
    ridge=0.003233351093667637,
    noise=2.9058065540337763e-06,
    input_scaling=1.9736136546215963,
    feedback_scaling=0.2014487190977695,
    W_in=W_in_fixed,
    W_fb=W_fb_fixed,
    bias=np.array([0.2]),
    seed=SEED,
)

model.fit(u_train, y_train, warmup=WARMUP)
warmup_pred = denormalize(model.predict(u_test[:WARMUP]))
auto_pred   = denormalize(model.predict_autonomous(len(u_test) - WARMUP))
full_pred   = np.concatenate([warmup_pred, auto_pred])
full_target = denormalize(y_test)

test_nmse = nmse(auto_pred, denormalize(y_test[WARMUP:]))
print(f"Test NMSE: {test_nmse:.6f}")

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write(f"Test NMSE: {test_nmse:.6f}\n")

n = len(full_target)
START = 50
fig, ax = plt.subplots(1, 1, figsize=((n - START) // 10, 6))
t = np.arange(START, n)
ax.axvspan(START, WARMUP, alpha=0.12, color="gray", label="Warmup")
ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1)
ax.plot(t, full_target[START:], label="Target",     color="steelblue")
ax.plot(t, full_pred[START:],   label="Prediction", color="tomato", linestyle="--")
ax.set_title(f"ESNCustomizable — Mackey-Glass refined (warmup={WARMUP}, test NMSE={test_nmse:.4f})")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.show()
