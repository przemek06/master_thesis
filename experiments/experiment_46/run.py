import sys
import os
print("Setting up environment...", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import optuna
from tasks.mackey_glass_multi import load
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_uniform_ring

HERE = os.path.dirname(__file__)
WARMUP = 1000
DIVERGENCE_WINDOW = 20
DIVERGENCE_THRESHOLD = 1.0
N_VAL = 5
N_TEST = 20
N_OPTUNA_TRIALS = 75
N_RESERVOIR = 400
N_INPUTS = 1
N_OUTPUTS = 1
SEED = 0


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


def eval_steps(model, series_list):
    scores = []
    for u, y in series_list:
        model.predict(u[:WARMUP])
        auto_pred = denormalize(model.predict_autonomous(len(u) - WARMUP))
        scores.append(steps_until_divergence(auto_pred, denormalize(y[WARMUP:])))
    return scores


def series_stats(scores):
    arr = np.array(scores)
    return {"scores": [int(s) for s in scores], "mean": float(arr.mean()), "std": float(arr.std())}


print("Loading data...", flush=True)
u_train, y_train, val_list, test_list = load(n_val=N_VAL, n_test=N_TEST)
print(f"Val: {len(val_list)}, Test: {len(test_list)}", flush=True)

rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))

optuna.logging.set_verbosity(optuna.logging.WARNING)

holder = {"model": None, "score": -np.inf, "W": None}


def objective(trial):
    r = trial.suggest_float("r", 0.5, 1.0)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_uniform_ring(r=r, size=size),
        N_RESERVOIR,
        trial.suggest_float("sparsity", 0.9, 0.99),
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
    score = float(np.mean(eval_steps(model, val_list)))
    if score > holder["score"]:
        holder["score"] = score
        holder["model"] = copy.deepcopy(model)
        holder["W"] = W
    return score


print("Optimizing...", flush=True)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

model = holder["model"]
W_best = holder["W"]

model.reset_state()
val_scores = eval_steps(model, val_list)
model.reset_state()
test_scores = eval_steps(model, test_list)
print(f"Val mean: {np.mean(val_scores):.1f}, Test mean: {np.mean(test_scores):.1f}", flush=True)

results = {
    "config": {
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "n_reservoir": N_RESERVOIR,
        "warmup": WARMUP,
        "seed": SEED,
        "divergence_window": DIVERGENCE_WINDOW,
        "divergence_threshold": DIVERGENCE_THRESHOLD,
    },
    "best_val_score_optuna": float(study.best_value),
    "val": series_stats(val_scores),
    "test": series_stats(test_scores),
    "best_params": {k: float(v) for k, v in study.best_params.items()},
}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write(f"Val mean:  {np.mean(val_scores):.1f} ± {np.std(val_scores):.1f}\n")
    f.write(f"Test mean: {np.mean(test_scores):.1f} ± {np.std(test_scores):.1f}\n")
    f.write(f"\nBest params:\n")
    for k, v in study.best_params.items():
        f.write(f"  {k}: {v}\n")

print("Plotting...", flush=True)

test_arr = np.array(test_scores)
median_idx = int(np.argsort(test_arr)[len(test_arr) // 2])
u_test_med, y_test_med = test_list[median_idx]
model.reset_state()
warmup_pred = denormalize(model.predict(u_test_med[:WARMUP]))
auto_pred_med = denormalize(model.predict_autonomous(len(u_test_med) - WARMUP))
full_pred = np.concatenate([warmup_pred, auto_pred_med])
full_target = denormalize(y_test_med)
med_steps = test_scores[median_idx]
n = len(full_target)
START = 50
fig, ax = plt.subplots(figsize=((n - START) // 10, 6))
t = np.arange(START, n)
ax.axvspan(START, WARMUP, alpha=0.12, color="gray", label="Warmup")
ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1)
ax.axvline(WARMUP + med_steps, color="red", linestyle="--", linewidth=1, label=f"Divergence at step {med_steps}")
ax.plot(t, full_target[START:], label="Target", color="steelblue")
ax.plot(t, full_pred[START:], label="Prediction", color="tomato", linestyle="--")
ax.set_title(f"ESNCustomizable (real eigenvalues) — Mackey-Glass\ntest mean={np.mean(test_scores):.1f}")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.close()

eigs = np.linalg.eigvals(W_best)
theta = np.linspace(0, 2 * np.pi, 500)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.6)
ax.set_aspect("equal")
ax.set_title("ESNCustomizable — Eigenvalue distribution of W (uniform ring)")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.close()

print("Done.", flush=True)
