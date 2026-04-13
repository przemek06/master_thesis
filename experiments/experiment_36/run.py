import sys
import os
print("Setting up environment...", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import torch
import optuna
from tasks.mackey_glass_refined import load
from models.esn_feedback import ESNFeedback

print("Environment setup complete.", flush=True)
HERE = os.path.dirname(__file__)
WARMUP = 1000
DIVERGENCE_WINDOW = 20
DIVERGENCE_THRESHOLD = 1.0

def denormalize(x):
    return np.arctanh(np.clip(x, -1 + 1e-7, 1 - 1e-7)) + 1

def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)

def steps_until_divergence(pred, target):
    n = len(pred)
    for i in range(DIVERGENCE_WINDOW, n + 1):
        window_nmse = nmse(pred[i - DIVERGENCE_WINDOW:i], target[i - DIVERGENCE_WINDOW:i])
        if window_nmse > DIVERGENCE_THRESHOLD:
            return i - DIVERGENCE_WINDOW
    return n

print("Loading data...", flush=True)
u_train, y_train, u_val, y_val, u_test, y_test = load()
print(f"Train: {len(u_train)}, Val: {len(u_val)}, Test: {len(u_test)}", flush=True)

N_RESERVOIR = 1000
N_INPUTS = 1
N_OUTPUTS = 1
SEED = 0

rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))


def objective(trial):
    model = ESNFeedback(
        n_inputs=N_INPUTS,
        n_reservoir=N_RESERVOIR,
        n_outputs=N_OUTPUTS,
        spectral_radius=trial.suggest_float("spectral_radius", 0.5, 1.0),
        sparsity=trial.suggest_float("sparsity", 0.95, 0.99),
        leaky_rate=trial.suggest_float("leaky_rate", 0.5, 1.0),
        ridge=trial.suggest_float("ridge", 1e-6, 1e-1, log=True),
        noise=trial.suggest_float("noise", 1e-6, 1e-2, log=True),
        input_scaling=trial.suggest_float("input_scaling", 0.1, 5.0),
        feedback_scaling=trial.suggest_float("feedback_scaling", 0.0, 1.0),
        W_in=W_in_fixed,
        W_fb=W_fb_fixed,
        seed=SEED,
    )
    model.fit(u_train, y_train, warmup=WARMUP)
    model.predict(u_val[:WARMUP])
    auto_pred = denormalize(model.predict_autonomous(len(u_val) - WARMUP))
    target = denormalize(y_val[WARMUP:])
    return steps_until_divergence(auto_pred, target)

print("Running Bayesian optimization...", flush=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=300, show_progress_bar=True)

best = study.best_params
print(f"Best val steps until divergence: {study.best_value}", flush=True)
print(f"Best params: {best}", flush=True)

model = ESNFeedback(
    n_inputs=N_INPUTS,
    n_reservoir=N_RESERVOIR,
    n_outputs=N_OUTPUTS,
    spectral_radius=best["spectral_radius"],
    sparsity=best["sparsity"],
    leaky_rate=best["leaky_rate"],
    ridge=best["ridge"],
    noise=best["noise"],
    input_scaling=best["input_scaling"],
    feedback_scaling=best["feedback_scaling"],
    W_in=W_in_fixed,
    W_fb=W_fb_fixed,
    seed=SEED,
)
print("Fitting on train...", flush=True)
model.fit(u_train, y_train, warmup=WARMUP)
print("Running autonomous prediction on test...", flush=True)
warmup_pred = denormalize(model.predict(u_test[:WARMUP]))
auto_pred   = denormalize(model.predict_autonomous(len(u_test) - WARMUP))
full_pred   = np.concatenate([warmup_pred, auto_pred])
full_target = denormalize(y_test)

test_steps = steps_until_divergence(auto_pred, denormalize(y_test[WARMUP:]))
print(f"Test steps until divergence: {test_steps}", flush=True)

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write(f"Best val steps until divergence: {study.best_value}\n")
    f.write(f"Test steps until divergence: {test_steps}\n")
    for k, v in best.items():
        f.write(f"{k}: {v}\n")

print("Plotting...", flush=True)
n = len(full_target)
START = 50
fig, ax = plt.subplots(1, 1, figsize=((n - START) // 10, 6))
t = np.arange(START, n)
ax.axvspan(START, WARMUP, alpha=0.12, color="gray", label="Warmup")
ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1)
ax.axvline(WARMUP + test_steps, color="red", linestyle="--", linewidth=1, label=f"Divergence at step {test_steps}")
ax.plot(t, full_target[START:], label="Target",     color="steelblue")
ax.plot(t, full_pred[START:],   label="Prediction", color="tomato", linestyle="--")
ax.set_title(f"ESNFeedback — Mackey-Glass refined (warmup={WARMUP}, test steps until divergence={test_steps})")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.show()
