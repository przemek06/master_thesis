import sys
import os
print("Setting up environment...", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import copy
import numpy as np
import matplotlib.pyplot as plt
import optuna
from tasks.narma30_batch import load
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
WARMUP = 200


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


N_RESERVOIR = 400
N_INPUTS = 1
N_OUTPUTS = 1
N_TRAIN = 10
N_VAL = 10
N_TEST = 10
SEED = 0

print("Loading data...", flush=True)
u_train, y_train, val_list, test_list = load(n_val=N_VAL, n_test=N_TEST, n_train=N_TRAIN)
print(f"Train: {N_TRAIN} series, Val: {len(val_list)}, Test: {len(test_list)}", flush=True)

rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))

holder = {"model": None, "score": np.inf}


def objective(trial):
    r_min = trial.suggest_float("r_min", 0.0, 0.95)
    r_max = trial.suggest_float("r_max", r_min, 1.0)
    alpha = trial.suggest_float("alpha", 0.1, 10.0)
    sparsity = trial.suggest_float("sparsity", 0.9, 0.99)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=r_min, r_max=r_max, alpha=alpha, size=size),
        N_RESERVOIR, sparsity, iterations=50, seed=SEED,
    )
    model = ESNCustomizable(
        n_inputs=N_INPUTS,
        n_reservoir=N_RESERVOIR,
        n_outputs=N_OUTPUTS,
        leaky_rate=trial.suggest_float("leaky_rate", 0.5, 1.0),
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
    if N_TRAIN == 1:
        model.fit(u_train, y_train, warmup=WARMUP)
    else:
        model.fit_batch(u_train, y_train, warmup=WARMUP)
    score = 0.0
    for u_val, y_val in val_list:
        model.predict(u_val[:WARMUP])
        pred = model.predict(u_val[WARMUP:])
        score += nmse(pred, y_val[WARMUP:])
    score /= len(val_list)
    if score < holder["score"]:
        holder["score"] = score
        holder["model"] = copy.deepcopy(model)
    return score


print("Running Bayesian optimization...", flush=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

best = study.best_params
print(f"Best val NMSE: {study.best_value}", flush=True)
print(f"Best params: {best}", flush=True)

model = holder["model"]
model.reset_state()

print("Evaluating on test series...", flush=True)
test_nmse_list = []
test_preds = []
for u_test, y_test in test_list:
    model.predict(u_test[:WARMUP])
    pred = model.predict(u_test[WARMUP:])
    test_nmse_list.append(nmse(pred, y_test[WARMUP:]))
    test_preds.append(pred)

test_nmse_arr = np.array(test_nmse_list)
mean_nmse = np.mean(test_nmse_arr)
std_nmse = np.std(test_nmse_arr)
min_nmse = np.min(test_nmse_arr)
max_nmse = np.max(test_nmse_arr)

print(f"Test NMSE — mean: {mean_nmse:.4f}, std: {std_nmse:.4f}, min: {min_nmse:.4f}, max: {max_nmse:.4f}", flush=True)

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write(f"Best val NMSE: {study.best_value:.4f}\n")
    f.write(f"Test mean NMSE: {mean_nmse:.4f}\n")
    f.write(f"Test std: {std_nmse:.4f}\n")
    f.write(f"Test min: {min_nmse:.4f}\n")
    f.write(f"Test max: {max_nmse:.4f}\n")
    f.write("\nPer-seed test NMSE:\n")
    for i, s in enumerate(test_nmse_list):
        f.write(f"  seed {N_TRAIN + N_VAL + i}: {s:.4f}\n")
    f.write("\nBest hyperparameters:\n")
    for k, v in best.items():
        f.write(f"  {k}: {v}\n")

best_idx = int(np.argmin(test_nmse_arr))
_, y_test_best = test_list[best_idx]
pred_best = test_preds[best_idx]
best_nmse = test_nmse_list[best_idx]

print("Plotting...", flush=True)
START = 50
t_eval = np.arange(WARMUP + START, WARMUP + len(pred_best))
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(t_eval, y_test_best[WARMUP + START:], label="Target", color="steelblue")
ax.plot(t_eval, pred_best[START:], label="Prediction", color="tomato", linestyle="--", alpha=0.8)
ax.set_xlabel("Step")
ax.set_ylabel("y")
ax.set_title(f"ESNCustomizable — NARMA-30 (best test series, seed {N_TRAIN + N_VAL + best_idx})\nNMSE={best_nmse:.4f}, mean={mean_nmse:.4f} ± {std_nmse:.4f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
seeds = [N_TRAIN + N_VAL + i for i in range(len(test_nmse_list))]
ax.bar(seeds, test_nmse_list, color="steelblue")
ax.axhline(mean_nmse, color="red", linestyle="--", linewidth=1.5, label=f"Mean = {mean_nmse:.4f}")
ax.axhline(mean_nmse - std_nmse, color="red", linestyle=":", linewidth=1, alpha=0.6)
ax.axhline(mean_nmse + std_nmse, color="red", linestyle=":", linewidth=1, alpha=0.6)
ax.set_xlabel("Test seed")
ax.set_ylabel("NMSE")
ax.set_title("ESNCustomizable — NARMA-30 batch, NMSE per test seed")
ax.set_xticks(seeds)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "test_nmse_bar.png"), dpi=150)
plt.close()
