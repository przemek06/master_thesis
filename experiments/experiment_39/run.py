import sys
import os
print("Setting up environment...", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import optuna
from tasks.mackey_glass_batch import load
from models.esn_feedback import ESNFeedback

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
u_train, y_train, val_list, test_list = load()
print(f"Train: {len(u_train)}, Val series: {len(val_list)}, Test series: {len(test_list)}", flush=True)

N_RESERVOIR = 400
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
    total_steps = 0
    for u_val, y_val in val_list:
        model.predict(u_val[:WARMUP])
        auto_pred = denormalize(model.predict_autonomous(len(u_val) - WARMUP))
        total_steps += steps_until_divergence(auto_pred, denormalize(y_val[WARMUP:]))
    return total_steps / len(val_list)


print("Running Bayesian optimization...", flush=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

best = study.best_params
print(f"Best val mean steps until divergence: {study.best_value}", flush=True)
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

print("Evaluating on test series...", flush=True)
test_steps_list = []
for u_test, y_test in test_list:
    model.predict(u_test[:WARMUP])
    auto_pred = denormalize(model.predict_autonomous(len(u_test) - WARMUP))
    test_steps_list.append(steps_until_divergence(auto_pred, denormalize(y_test[WARMUP:])))

test_steps_arr = np.array(test_steps_list)
mean_steps = np.mean(test_steps_arr)
std_steps = np.std(test_steps_arr)
min_steps = np.min(test_steps_arr)
max_steps = np.max(test_steps_arr)

print(f"Test steps — mean: {mean_steps:.1f}, std: {std_steps:.1f}, min: {min_steps}, max: {max_steps}", flush=True)

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write(f"Best val mean steps until divergence: {study.best_value:.1f}\n")
    f.write(f"Test mean steps until divergence: {mean_steps:.1f}\n")
    f.write(f"Test std: {std_steps:.1f}\n")
    f.write(f"Test min: {min_steps}\n")
    f.write(f"Test max: {max_steps}\n")
    f.write("\nPer-seed test steps:\n")
    for i, s in enumerate(test_steps_list):
        f.write(f"  seed {6 + i}: {s}\n")
    f.write("\nBest hyperparameters:\n")
    for k, v in best.items():
        f.write(f"  {k}: {v}\n")

median_idx = int(np.argsort(test_steps_arr)[len(test_steps_arr) // 2])
u_test_med, y_test_med = test_list[median_idx]
warmup_pred = denormalize(model.predict(u_test_med[:WARMUP]))
auto_pred_med = denormalize(model.predict_autonomous(len(u_test_med) - WARMUP))
full_pred = np.concatenate([warmup_pred, auto_pred_med])
full_target = denormalize(y_test_med)
med_steps = test_steps_list[median_idx]

print("Plotting...", flush=True)
n = len(full_target)
START = 50
fig, ax = plt.subplots(figsize=((n - START) // 10, 6))
t = np.arange(START, n)
ax.axvspan(START, WARMUP, alpha=0.12, color="gray", label="Warmup")
ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1)
ax.axvline(WARMUP + med_steps, color="red", linestyle="--", linewidth=1, label=f"Divergence at step {med_steps}")
ax.plot(t, full_target[START:], label="Target", color="steelblue")
ax.plot(t, full_pred[START:], label="Prediction", color="tomato", linestyle="--")
ax.set_title(f"ESNFeedback — Mackey-Glass batch (median test series, seed {6 + median_idx})\nmean={mean_steps:.1f} ± {std_steps:.1f} steps")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
seeds = [6 + i for i in range(len(test_steps_list))]
ax.bar(seeds, test_steps_list, color="steelblue")
ax.axhline(mean_steps, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_steps:.1f}")
ax.axhline(mean_steps - std_steps, color="red", linestyle=":", linewidth=1, alpha=0.6)
ax.axhline(mean_steps + std_steps, color="red", linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel("Test seed")
ax.set_ylabel("Steps until divergence")
ax.set_title("ESNFeedback — Steps until divergence per test seed")
ax.set_xticks(seeds)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "test_steps_bar.png"), dpi=150)
plt.close()
