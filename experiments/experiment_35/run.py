import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import optuna
from tasks.mackey_glass_refined import load
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

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
SPARSITY = 0.9697222820893018

rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))


def objective(trial):
    r_min = trial.suggest_float("r_min", 0.0, 0.95)
    r_max = trial.suggest_float("r_max", r_min, 1.0)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=r_min, r_max=r_max,
                                                alpha=trial.suggest_float("alpha", 0.1, 10.0),
                                                size=size),
        N_RESERVOIR,
        SPARSITY,
        iterations=50,
        seed=SEED,
    )
    model = ESNCustomizable(
        n_inputs=N_INPUTS,
        n_reservoir=N_RESERVOIR,
        n_outputs=N_OUTPUTS,
        leaky_rate=trial.suggest_float("leaky_rate", 0.1, 1.0),
        ridge=trial.suggest_float("ridge", 1e-6, 1e-1, log=True),
        noise=trial.suggest_float("noise", 1e-6, 1e-2, log=True),
        input_scaling=trial.suggest_float("input_scaling", 0.1, 5.0),
        feedback_scaling=trial.suggest_float("feedback_scaling", 0.0, 1.0),
        W_in=W_in_fixed,
        W=W,
        W_fb=W_fb_fixed,
        bias=np.array([0.2]),
        seed=SEED,
    )
    model.fit(u_train, y_train, warmup=WARMUP)
    model.predict(u_val[:WARMUP])
    auto_pred = denormalize(model.predict_autonomous(len(u_val) - WARMUP))
    target = denormalize(y_val[WARMUP:])
    return nmse(auto_pred, target)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=300, show_progress_bar=True)

best = study.best_params
print(f"Best val NMSE: {study.best_value:.6f}")
print(f"Best params: {best}")

W = generate_isospectral_sparse_matrix(
    lambda size: sample_eigenvalues_ginibre(r_min=best["r_min"], r_max=best["r_max"],
                                            alpha=best["alpha"], size=size),
    N_RESERVOIR,
    SPARSITY,
    iterations=50,
    seed=SEED,
)

model = ESNCustomizable(
    n_inputs=N_INPUTS,
    n_reservoir=N_RESERVOIR,
    n_outputs=N_OUTPUTS,
    leaky_rate=best["leaky_rate"],
    ridge=best["ridge"],
    noise=best["noise"],
    input_scaling=best["input_scaling"],
    feedback_scaling=best["feedback_scaling"],
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
    f.write(f"Best val NMSE: {study.best_value:.6f}\n")
    f.write(f"Test NMSE: {test_nmse:.6f}\n")
    for k, v in best.items():
        f.write(f"{k}: {v}\n")

n = len(full_target)
START = 50
fig, ax = plt.subplots(1, 1, figsize=((n - START) // 10, 6))
t = np.arange(START, n)
ax.axvspan(START, WARMUP, alpha=0.12, color="gray", label="Warmup")
ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1)
ax.plot(t, full_target[START:], label="Target",     color="steelblue")
ax.plot(t, full_pred[START:],   label="Prediction", color="tomato", linestyle="--")
ax.set_title(f"ESNCustomizable (Stiefel W) — Mackey-Glass refined (warmup={WARMUP}, test NMSE={test_nmse:.4f})")
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
