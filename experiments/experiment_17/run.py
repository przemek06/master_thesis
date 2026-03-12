import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import optuna
from tasks.lorenz import load
from models.esn import ESN

HERE = os.path.dirname(__file__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_RESERVOIR = 200
WARMUP = 100


def nmse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true)


u_train, y_train, u_val, y_val, u_test, y_test = load()


def objective(trial):
    params = {
        "spectral_radius": trial.suggest_float("spectral_radius", 0.1, 1.5),
        "sparsity":        trial.suggest_float("sparsity", 0.5, 0.99),
        "input_scaling":   trial.suggest_float("input_scaling", 0.1, 2.0),
        "leaky_rate":      trial.suggest_float("leaky_rate", 0.1, 1.0),
        "ridge":           trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
    }
    esn = ESN(n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1, seed=0, **params)
    esn.fit(u_train, y_train, warmup=WARMUP)
    y_pred = esn.predict_autonomous(u_val[:WARMUP], len(u_val) - WARMUP)
    return nmse(y_val[WARMUP:], y_pred)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best val NMSE: {study.best_value:.6f}")
print(f"Best params:   {study.best_params}")

best = study.best_params
esn = ESN(n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1, seed=0, **best)
esn.fit(u_train, y_train, warmup=WARMUP)
y_pred = esn.predict_autonomous(u_test[:WARMUP], len(u_test) - WARMUP)

test_nmse = nmse(y_test[WARMUP:], y_pred)
print(f"Test NMSE:     {test_nmse:.6f}")

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write(f"Val NMSE:  {study.best_value:.6f}\n")
    f.write(f"Test NMSE: {test_nmse:.6f}\n")
    f.write(f"Best params: {study.best_params}\n")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

for i, ax in enumerate(axes):
    if i == 0:
        warmup_preds = esn.predict(u_test[:WARMUP])
        full_pred = np.concatenate([warmup_preds, y_pred])
        ax.axvspan(0, WARMUP, alpha=0.15, color="gray", label="Warmup")
        ax.plot(np.concatenate([y_test[:WARMUP], y_test[WARMUP:WARMUP + 200]])[:WARMUP + 200], label="Target", color="steelblue")
        ax.plot(full_pred[:WARMUP + 200], label="Prediction", color="tomato", linestyle="--")
        ax.set_title(f"Timesteps 0–{WARMUP + 200} (warmup shaded)")
    else:
        start = i * 400
        end = start + 200
        ax.plot(y_test[WARMUP + start:WARMUP + end], label="Target", color="steelblue")
        ax.plot(y_pred[start:end], label="Prediction", color="tomato", linestyle="--")
        ax.set_title(f"Prediction timesteps {start}–{end}")
    ax.legend(loc="upper right")

fig.suptitle("ESN — Lorenz predictions vs target")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.show()

trials_df = study.trials_dataframe()
plt.figure(figsize=(8, 4))
plt.plot(trials_df["number"], trials_df["value"], alpha=0.4, label="Trial val NMSE")
plt.plot(trials_df["number"], trials_df["value"].cummin(), color="tomato", label="Best so far")
plt.xlabel("Trial")
plt.ylabel("Val NMSE")
plt.title("Hyperparameter optimization — ESN Lorenz")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "optimization.png"), dpi=150)
plt.show()

W_np = esn.W.cpu().numpy()
eigenvalues = np.linalg.eigvals(W_np)
theta = np.linspace(0, 2 * np.pi, 500)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
ax.scatter(eigenvalues.real, eigenvalues.imag, s=10, alpha=0.6)
ax.set_aspect("equal")
ax.set_title("Eigenvalues of W — ESN Lorenz")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.show()
