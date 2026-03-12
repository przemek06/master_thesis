import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import optuna
from tasks.narma30 import load
from models.custom_esn_2 import CustomESN2

HERE = os.path.dirname(__file__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_RESERVOIR = 200
ITERATIONS = 200
THRESHOLD = 0.01


def nmse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true)


u_train, y_train, u_val, y_val, u_test, y_test = load()


def objective(trial):
    params = {
        "r":             trial.suggest_float("r", 0.1, 1.5),
        "alpha":         trial.suggest_float("alpha", 0.1, 10.0),
        "beta":          trial.suggest_float("beta", 0.1, 10.0),
        "mu":            0.0,
        "kappa":         0.0,
        "sparsity":      trial.suggest_float("sparsity", 0.5, 0.99),
        "input_scaling": trial.suggest_float("input_scaling", 0.1, 2.0),
        "ridge":         trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
    }
    model = CustomESN2(n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1, seed=0, iterations=ITERATIONS, threshold=THRESHOLD, **params)
    model.fit(u_train, y_train, warmup=100)
    y_pred = model.predict(u_val)
    return nmse(y_val, y_pred)


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best val NMSE: {study.best_value:.6f}")
print(f"Best params:   {study.best_params}")

best = study.best_params
model = CustomESN2(n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1, seed=0, iterations=ITERATIONS, threshold=THRESHOLD, mu=0.0, kappa=0.0, **best)
model.fit(u_train, y_train, warmup=100)
y_pred = model.predict(u_test)

print(f"Test NMSE:     {nmse(y_test, y_pred):.6f}")

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
for i, ax in enumerate(axes):
    start = i * 100
    end = start + 200
    ax.plot(y_test[start:end], label="Target", color="steelblue")
    ax.plot(y_pred[start:end], label="Prediction", color="tomato", linestyle="--")
    ax.set_title(f"Timesteps {start}–{end}")
    ax.legend(loc="upper right")

fig.suptitle("CustomESN2 (tuned) — NARMA-30 predictions vs target")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.show()

trials_df = study.trials_dataframe()
plt.figure(figsize=(8, 4))
plt.plot(trials_df["number"], trials_df["value"], alpha=0.4, label="Trial val NMSE")
plt.plot(trials_df["number"], trials_df["value"].cummin(), color="tomato", label="Best so far")
plt.xlabel("Trial")
plt.ylabel("Val NMSE")
plt.title("Hyperparameter optimization — CustomESN2 NARMA-30")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "optimization.png"), dpi=150)
plt.show()
