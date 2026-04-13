import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
import optuna
from tasks.mackey_glass_batch import load
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

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


u_train, y_train, val_list, test_list = load()

N_RESERVOIR = 400
N_INPUTS = 1
N_OUTPUTS = 1
SEED = 0

rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))


def objective(trial):
    r_min = trial.suggest_float("r_min", 0.0, 0.95)
    r_max = trial.suggest_float("r_max", r_min, 1.0)
    sparsity = trial.suggest_float("sparsity", 0.9, 0.99)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=r_min, r_max=r_max,
                                                alpha=trial.suggest_float("alpha", 0.1, 10.0),
                                                size=size),
        N_RESERVOIR,
        sparsity,
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
    total_steps = 0
    for u_val, y_val in val_list:
        model.predict(u_val[:WARMUP])
        auto_pred = denormalize(model.predict_autonomous(len(u_val) - WARMUP))
        total_steps += steps_until_divergence(auto_pred, denormalize(y_val[WARMUP:]))
    return total_steps / len(val_list)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

best = study.best_params
print(f"Best val mean steps until divergence: {study.best_value}")
print(f"Best params: {best}")

W = generate_isospectral_sparse_matrix(
    lambda size: sample_eigenvalues_ginibre(r_min=best["r_min"], r_max=best["r_max"],
                                            alpha=best["alpha"], size=size),
    N_RESERVOIR,
    best["sparsity"],
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

print(f"Test steps — mean: {mean_steps:.1f}, std: {std_steps:.1f}, min: {min_steps}, max: {max_steps}")

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

n = len(full_target)
START = 50
fig, ax = plt.subplots(figsize=((n - START) // 10, 6))
t = np.arange(START, n)
ax.axvspan(START, WARMUP, alpha=0.12, color="gray", label="Warmup")
ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1)
ax.axvline(WARMUP + med_steps, color="red", linestyle="--", linewidth=1, label=f"Divergence at step {med_steps}")
ax.plot(t, full_target[START:], label="Target", color="steelblue")
ax.plot(t, full_pred[START:], label="Prediction", color="tomato", linestyle="--")
ax.set_title(f"ESNCustomizable (Stiefel W) — Mackey-Glass batch (median test series, seed {6 + median_idx})\nmean={mean_steps:.1f} ± {std_steps:.1f} steps")
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
ax.set_title("ESNCustomizable (Stiefel W) — Steps until divergence per test seed")
ax.set_xticks(seeds)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "test_steps_bar.png"), dpi=150)
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
