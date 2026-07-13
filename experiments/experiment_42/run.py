import sys
import os
print("Setting up environment...", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import optuna
from tasks.mackey_glass_multi import load
from models.esn_feedback import ESNFeedback
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
WARMUP = 1000
DIVERGENCE_WINDOW = 20
DIVERGENCE_THRESHOLD = 1.0
N_VAL = 10
N_TEST = 50
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


def eval_model(model, test_list):
    steps = []
    for u_test, y_test in test_list:
        model.predict(u_test[:WARMUP])
        auto_pred = denormalize(model.predict_autonomous(len(u_test) - WARMUP))
        steps.append(steps_until_divergence(auto_pred, denormalize(y_test[WARMUP:])))
    return steps


def val_score(model, val_list):
    total = 0
    for u_val, y_val in val_list:
        model.predict(u_val[:WARMUP])
        auto_pred = denormalize(model.predict_autonomous(len(u_val) - WARMUP))
        total += steps_until_divergence(auto_pred, denormalize(y_val[WARMUP:]))
    return total / len(val_list)


print("Loading data...", flush=True)
u_train, y_train, val_list, test_list = load(n_val=N_VAL, n_test=N_TEST)
print(f"Train: {len(u_train)}, Val: {len(val_list)}, Test: {len(test_list)}", flush=True)

rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))


print("Optimizing ESNFeedback...", flush=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_a(trial):
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
    return val_score(model, val_list)


study_a = optuna.create_study(direction="maximize")
study_a.optimize(objective_a, n_trials=100, show_progress_bar=True)
best_a = study_a.best_params
print(f"ESNFeedback best val: {study_a.best_value:.1f}", flush=True)

model_a = ESNFeedback(
    n_inputs=N_INPUTS,
    n_reservoir=N_RESERVOIR,
    n_outputs=N_OUTPUTS,
    spectral_radius=best_a["spectral_radius"],
    sparsity=best_a["sparsity"],
    leaky_rate=best_a["leaky_rate"],
    ridge=best_a["ridge"],
    noise=best_a["noise"],
    input_scaling=best_a["input_scaling"],
    feedback_scaling=best_a["feedback_scaling"],
    W_in=W_in_fixed,
    W_fb=W_fb_fixed,
    seed=SEED,
)
model_a.fit(u_train, y_train, warmup=WARMUP)
test_steps_a = eval_model(model_a, test_list)
arr_a = np.array(test_steps_a)
print(f"ESNFeedback test — mean: {arr_a.mean():.1f}, std: {arr_a.std():.1f}", flush=True)


print("Optimizing ESNCustomizable...", flush=True)


def objective_b(trial):
    r_min = trial.suggest_float("r_min", 0.0, 0.95)
    r_max = trial.suggest_float("r_max", r_min, 1.0)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(
            r_min=r_min, r_max=r_max,
            alpha=trial.suggest_float("alpha", 0.1, 10.0),
            size=size,
        ),
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
    return val_score(model, val_list)


study_b = optuna.create_study(direction="maximize")
study_b.optimize(objective_b, n_trials=100, show_progress_bar=True)
best_b = study_b.best_params
print(f"ESNCustomizable best val: {study_b.best_value:.1f}", flush=True)

W_best = generate_isospectral_sparse_matrix(
    lambda size: sample_eigenvalues_ginibre(
        r_min=best_b["r_min"], r_max=best_b["r_max"],
        alpha=best_b["alpha"], size=size,
    ),
    N_RESERVOIR,
    best_b["sparsity"],
    iterations=50,
    seed=SEED,
)

model_b = ESNCustomizable(
    n_inputs=N_INPUTS,
    n_reservoir=N_RESERVOIR,
    n_outputs=N_OUTPUTS,
    leaky_rate=best_b["leaky_rate"],
    ridge=best_b["ridge"],
    noise=best_b["noise"],
    input_scaling=best_b["input_scaling"],
    feedback_scaling=best_b["feedback_scaling"],
    W_in=W_in_fixed,
    W=W_best,
    W_fb=W_fb_fixed,
    bias=np.array([0.2]),
    seed=SEED,
)
model_b.fit(u_train, y_train, warmup=WARMUP)
test_steps_b = eval_model(model_b, test_list)
arr_b = np.array(test_steps_b)
print(f"ESNCustomizable test — mean: {arr_b.mean():.1f}, std: {arr_b.std():.1f}", flush=True)


stat, pvalue = stats.wilcoxon(arr_a, arr_b)
better = "ESNCustomizable" if arr_b.mean() > arr_a.mean() else "ESNFeedback"
conclusion = f"{better} has higher mean steps. p={pvalue:.4f} ({'significant' if pvalue < 0.05 else 'not significant'} at alpha=0.05)"
print(f"Wilcoxon: W={stat:.1f}, p={pvalue:.4f} — {conclusion}", flush=True)


with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write("=== ESNFeedback ===\n")
    f.write(f"Best val mean steps: {study_a.best_value:.1f}\n")
    f.write(f"Test mean: {arr_a.mean():.1f}, std: {arr_a.std():.1f}, min: {arr_a.min()}, max: {arr_a.max()}\n")
    f.write("Per-test steps:\n")
    for i, s in enumerate(test_steps_a):
        f.write(f"  test_{i}: {s}\n")
    f.write("Best hyperparameters:\n")
    for k, v in best_a.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n=== ESNCustomizable ===\n")
    f.write(f"Best val mean steps: {study_b.best_value:.1f}\n")
    f.write(f"Test mean: {arr_b.mean():.1f}, std: {arr_b.std():.1f}, min: {arr_b.min()}, max: {arr_b.max()}\n")
    f.write("Per-test steps:\n")
    for i, s in enumerate(test_steps_b):
        f.write(f"  test_{i}: {s}\n")
    f.write("Best hyperparameters:\n")
    for k, v in best_b.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n=== Statistical Test ===\n")
    f.write("Wilcoxon signed-rank test (paired, two-sided)\n")
    f.write(f"W = {stat:.1f}, p = {pvalue:.4f}\n")
    f.write(f"{conclusion}\n")


def prediction_plot(model, test_list, steps_list, arr, title, path):
    median_idx = int(np.argsort(arr)[len(arr) // 2])
    u_test_med, y_test_med = test_list[median_idx]
    warmup_pred = denormalize(model.predict(u_test_med[:WARMUP]))
    auto_pred_med = denormalize(model.predict_autonomous(len(u_test_med) - WARMUP))
    full_pred = np.concatenate([warmup_pred, auto_pred_med])
    full_target = denormalize(y_test_med)
    med_steps = steps_list[median_idx]
    n = len(full_target)
    START = 50
    fig, ax = plt.subplots(figsize=((n - START) // 10, 6))
    t = np.arange(START, n)
    ax.axvspan(START, WARMUP, alpha=0.12, color="gray", label="Warmup")
    ax.axvline(WARMUP, color="gray", linestyle=":", linewidth=1)
    ax.axvline(WARMUP + med_steps, color="red", linestyle="--", linewidth=1, label=f"Divergence at step {med_steps}")
    ax.plot(t, full_target[START:], label="Target", color="steelblue")
    ax.plot(t, full_pred[START:], label="Prediction", color="tomato", linestyle="--")
    ax.set_title(f"{title} (median test, idx {median_idx})\nmean={arr.mean():.1f} ± {arr.std():.1f} steps")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


print("Plotting predictions...", flush=True)
prediction_plot(model_a, test_list, test_steps_a, arr_a, "ESNFeedback — Mackey-Glass", os.path.join(HERE, "predictions_a.png"))
prediction_plot(model_b, test_list, test_steps_b, arr_b, "ESNCustomizable — Mackey-Glass", os.path.join(HERE, "predictions_b.png"))

print("Plotting test steps bar charts...", flush=True)
test_indices = list(range(N_TEST))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
ax1.bar(test_indices, test_steps_a, color="steelblue")
ax1.axhline(arr_a.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {arr_a.mean():.1f}")
ax1.axhline(arr_a.mean() - arr_a.std(), color="red", linestyle=":", linewidth=1, alpha=0.6)
ax1.axhline(arr_a.mean() + arr_a.std(), color="red", linestyle=":", linewidth=1, alpha=0.6)
ax1.set_ylabel("Steps until divergence")
ax1.set_title("ESNFeedback — Steps until divergence per test set")
ax1.legend()

ax2.bar(test_indices, test_steps_b, color="darkorange")
ax2.axhline(arr_b.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {arr_b.mean():.1f}")
ax2.axhline(arr_b.mean() - arr_b.std(), color="red", linestyle=":", linewidth=1, alpha=0.6)
ax2.axhline(arr_b.mean() + arr_b.std(), color="red", linestyle=":", linewidth=1, alpha=0.6)
ax2.set_xlabel("Test set index")
ax2.set_ylabel("Steps until divergence")
ax2.set_title("ESNCustomizable — Steps until divergence per test set")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(HERE, "test_steps_bar.png"), dpi=150)
plt.close()

print("Plotting score distributions...", flush=True)
fig, ax = plt.subplots(figsize=(10, 5))
x_min = min(arr_a.min(), arr_b.min()) - 50
x_max = max(arr_a.max(), arr_b.max()) + 50
x = np.linspace(x_min, x_max, 300)
for arr, label, color in [(arr_a, "ESNFeedback", "steelblue"), (arr_b, "ESNCustomizable", "darkorange")]:
    kde = gaussian_kde(arr, bw_method="scott")
    ax.plot(x, kde(x), label=f"{label} (mean={arr.mean():.1f})", color=color, linewidth=2)
    ax.fill_between(x, kde(x), alpha=0.15, color=color)
ax.set_xlabel("Steps until divergence")
ax.set_ylabel("Density")
ax.set_title(f"Test score distribution — Wilcoxon p={pvalue:.4f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "test_score_distribution.png"), dpi=150)
plt.close()

print("Plotting eigenvalues...", flush=True)
eigs = np.linalg.eigvals(W_best)
theta = np.linspace(0, 2 * np.pi, 500)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.6)
ax.set_aspect("equal")
ax.set_title("ESNCustomizable — Eigenvalue distribution of W")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.close()

print("Done.", flush=True)
