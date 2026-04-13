import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from tasks.mackey_glass_refined import load
from models.esn_customizable import ESNCustomizable
from generate import precompute_sparse_schur, generate_sparse_schur
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
WARMUP = 1000

def denormalize(x):
    return np.arctanh(np.clip(x, -1 + 1e-7, 1 - 1e-7)) + 1

def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)

u_train, y_train, u_val, y_val, u_test, y_test = load()

N_RESERVOIR = 400
N_INPUTS = 1
N_OUTPUTS = 1
SEED = 0
SPARSITY = 0.9697222820893018

rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))

sparsity_mask = rng.random((N_RESERVOIR, N_RESERVOIR)) > SPARSITY
T_template, Z_template = precompute_sparse_schur(sparsity_mask, seed=SEED)
eigenvalues = sample_eigenvalues_ginibre(r_min=0.85, r_max=0.9, alpha=1.0, size=N_RESERVOIR)
W = generate_sparse_schur(T_template, Z_template, sparsity_mask, eigenvalues)

model = ESNCustomizable(
    n_inputs=N_INPUTS,
    n_reservoir=N_RESERVOIR,
    n_outputs=N_OUTPUTS,
    leaky_rate=0.7987598599205421,
    ridge=0.003233351093667637,
    noise=2.9058065540337763e-06,
    input_scaling=1.9736136546215963,
    feedback_scaling=0.2014487190977695,
    W_in=W_in_fixed,
    W=W,
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
ax.set_title(f"ESNCustomizable (Ginibre W) — Mackey-Glass refined (warmup={WARMUP}, test NMSE={test_nmse:.4f})")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.close()

eigs = np.linalg.eigvals(W)
theta = np.linspace(0, 2 * np.pi, 500)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.6)
ax.set_aspect("equal")
ax.set_title("Eigenvalue distribution of W")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.close()

W_np = np.abs(W)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(W_np, aspect="auto", cmap="viridis", interpolation="none")
plt.colorbar(im, ax=ax)
ax.set_title("Heatmap of |W|")
ax.set_xlabel("j")
ax.set_ylabel("i")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "heatmap.png"), dpi=150)
plt.close()
