import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import optuna
from tasks.lorenz import load
from models.custom_esn_3 import CustomESN3

HERE = os.path.dirname(__file__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_RESERVOIR = 200
ITERATIONS = 200
THRESHOLD = 0.01
WARMUP = 100


def nmse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true)


u_train, y_train, u_val, y_val, u_test, y_test = load()


def objective(trial):
    params = {
        "r_min":         trial.suggest_float("r_min", 0.0, 0.5),
        "r_max":         trial.suggest_float("r_max", 0.5, 1.5),
        "alpha":         trial.suggest_float("alpha", 0.1, 10.0),
        "sparsity":      trial.suggest_float("sparsity", 0.5, 0.99),
        "input_scaling": trial.suggest_float("input_scaling", 0.1, 2.0),
        "ridge":         trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
    }
    model = CustomESN3(n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1, seed=0, iterations=ITERATIONS, threshold=THRESHOLD, **params)
    model.fit(u_train, y_train, warmup=WARMUP)
    y_pred = model.predict_autonomous(u_val[:WARMUP], len(u_val) - WARMUP)
    return nmse(y_val[WARMUP:], y_pred)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best val NMSE: {study.best_value:.6f}")
print(f"Best params:   {study.best_params}")

best = study.best_params
model = CustomESN3(n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1, seed=0, iterations=ITERATIONS, threshold=THRESHOLD, **best)
model.fit(u_train, y_train, warmup=WARMUP)
y_pred = model.predict_autonomous(u_test[:WARMUP], len(u_test) - WARMUP)

print(f"Test NMSE:     {nmse(y_test[WARMUP:], y_pred):.6f}")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)

for i, ax in enumerate(axes):
    if i == 0:
        warmup_preds = model.predict(u_test[:WARMUP])
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

fig.suptitle("CustomESN3 — Lorenz predictions vs target")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.show()

trials_df = study.trials_dataframe()
plt.figure(figsize=(8, 4))
plt.plot(trials_df["number"], trials_df["value"], alpha=0.4, label="Trial val NMSE")
plt.plot(trials_df["number"], trials_df["value"].cummin(), color="tomato", label="Best so far")
plt.xlabel("Trial")
plt.ylabel("Val NMSE")
plt.title("Hyperparameter optimization — CustomESN3 Lorenz")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "optimization.png"), dpi=150)
plt.show()

W_np = model.W.cpu().numpy()
eigenvalues = np.linalg.eigvals(W_np)
theta = np.linspace(0, 2 * np.pi, 500)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
ax.scatter(eigenvalues.real, eigenvalues.imag, s=10, alpha=0.6)
ax.set_aspect("equal")
ax.set_title("Eigenvalues of W — CustomESN3 Lorenz")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.show()
