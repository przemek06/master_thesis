import sys
import os
print("Setting up environment...", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import copy
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import optuna
from tasks.narma30_batch import load
from models.esn_feedback import ESNFeedback
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
WARMUP = 200
N_RUNS = 10
N_TRAIN = 10
N_VAL = 10
N_TEST = 10
N_OPTUNA_TRIALS = 100
N_RESERVOIR = 400
N_INPUTS = 1
N_OUTPUTS = 1
SEED = 0
OPTUNA_SEED = 0


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def eval_nmse(model, series_list):
    scores = []
    for u, y in series_list:
        model.predict(u[:WARMUP])
        pred = model.predict(u[WARMUP:])
        scores.append(nmse(pred, y[WARMUP:]))
    return scores


def series_stats(scores):
    arr = np.array(scores)
    return {"scores": [float(s) for s in scores], "mean": float(arr.mean()), "std": float(arr.std())}


rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))

optuna.logging.set_verbosity(optuna.logging.WARNING)

run_results = []
stored_models_a = []
stored_models_b = []
stored_W_best = []
stored_test_lists = []

for run_idx in range(N_RUNS):
    data_offset = run_idx * (N_TRAIN + N_VAL + N_TEST)
    print(f"\n=== Run {run_idx + 1}/{N_RUNS} (data_offset={data_offset}) ===", flush=True)

    u_train, y_train, val_list, test_list = load(n_val=N_VAL, n_test=N_TEST, n_train=N_TRAIN, seed_offset=data_offset)
    stored_test_lists.append(test_list)

    print("  Optimizing ESNFeedback...", flush=True)

    holder_a = {"model": None, "score": np.inf}

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
        model.fit_batch(u_train, y_train, warmup=WARMUP)
        score = float(np.mean(eval_nmse(model, val_list)))
        if score < holder_a["score"]:
            holder_a["score"] = score
            holder_a["model"] = copy.deepcopy(model)
        return score

    study_a = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study_a.optimize(objective_a, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    best_a = study_a.best_params

    model_a = holder_a["model"]
    model_a.reset_state()
    val_scores_a = eval_nmse(model_a, val_list)
    model_a.reset_state()
    test_scores_a = eval_nmse(model_a, test_list)
    print(f"  ESNFeedback     val mean: {np.mean(val_scores_a):.4f}, test mean: {np.mean(test_scores_a):.4f}", flush=True)

    print("  Optimizing ESNCustomizable...", flush=True)

    holder_b = {"model": None, "score": np.inf, "W": None}

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
        model.fit_batch(u_train, y_train, warmup=WARMUP)
        score = float(np.mean(eval_nmse(model, val_list)))
        if score < holder_b["score"]:
            holder_b["score"] = score
            holder_b["model"] = copy.deepcopy(model)
            holder_b["W"] = W
        return score

    study_b = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study_b.optimize(objective_b, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    best_b = study_b.best_params

    model_b = holder_b["model"]
    W_best = holder_b["W"]
    model_b.reset_state()
    val_scores_b = eval_nmse(model_b, val_list)
    model_b.reset_state()
    test_scores_b = eval_nmse(model_b, test_list)
    print(f"  ESNCustomizable val mean: {np.mean(val_scores_b):.4f}, test mean: {np.mean(test_scores_b):.4f}", flush=True)

    run_results.append({
        "run_index": run_idx,
        "data_offset": data_offset,
        "esn_feedback": {
            "best_val_score_optuna": float(study_a.best_value),
            "val": series_stats(val_scores_a),
            "test": series_stats(test_scores_a),
            "best_params": {k: float(v) for k, v in best_a.items()},
        },
        "esn_customizable": {
            "best_val_score_optuna": float(study_b.best_value),
            "val": series_stats(val_scores_b),
            "test": series_stats(test_scores_b),
            "best_params": {k: float(v) for k, v in best_b.items()},
        },
    })
    stored_models_a.append(model_a)
    stored_models_b.append(model_b)
    stored_W_best.append(W_best)

run_means_a = np.array([r["esn_feedback"]["test"]["mean"] for r in run_results])
run_means_b = np.array([r["esn_customizable"]["test"]["mean"] for r in run_results])
t_stat, p_value = stats.ttest_rel(run_means_a, run_means_b)
better = "esn_customizable" if run_means_b.mean() < run_means_a.mean() else "esn_feedback"

results = {
    "config": {
        "n_runs": N_RUNS,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "n_reservoir": N_RESERVOIR,
        "warmup": WARMUP,
        "seed": SEED,
        "optuna_seed": OPTUNA_SEED,
    },
    "runs": run_results,
    "summary": {
        "esn_feedback": {
            "run_means": run_means_a.tolist(),
            "mean_of_means": float(run_means_a.mean()),
            "std_of_means": float(run_means_a.std()),
        },
        "esn_customizable": {
            "run_means": run_means_b.tolist(),
            "mean_of_means": float(run_means_b.mean()),
            "std_of_means": float(run_means_b.std()),
        },
        "statistical_test": {
            "type": "paired_t_test",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "better_model": better,
        },
    },
}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write("=== Summary ===\n")
    f.write(f"ESNFeedback     test mean of means: {run_means_a.mean():.4f} ± {run_means_a.std():.4f}\n")
    f.write(f"ESNCustomizable test mean of means: {run_means_b.mean():.4f} ± {run_means_b.std():.4f}\n")
    f.write(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05)\n")
    f.write(f"Better model: {better}\n")
    f.write("\n=== Per-run results ===\n")
    for r in run_results:
        f.write(f"\nRun {r['run_index']} (data_offset={r['data_offset']})\n")
        f.write(f"  ESNFeedback     val mean: {r['esn_feedback']['val']['mean']:.4f}, test mean: {r['esn_feedback']['test']['mean']:.4f}\n")
        f.write(f"  ESNCustomizable val mean: {r['esn_customizable']['val']['mean']:.4f}, test mean: {r['esn_customizable']['test']['mean']:.4f}\n")

print("Plotting...", flush=True)

median_run_a = int(np.argsort(run_means_a)[N_RUNS // 2])
median_run_b = int(np.argsort(run_means_b)[N_RUNS // 2])


def prediction_plot(model, test_list, test_scores, run_mean, title, path):
    arr = np.array(test_scores)
    median_idx = int(np.argsort(arr)[len(arr) // 2])
    u_test_med, y_test_med = test_list[median_idx]
    model.predict(u_test_med[:WARMUP])
    pred_med = model.predict(u_test_med[WARMUP:])
    med_nmse = test_scores[median_idx]
    START = 50
    t = np.arange(WARMUP + START, WARMUP + len(pred_med))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, y_test_med[WARMUP + START:], label="Target", color="steelblue")
    ax.plot(t, pred_med[START:], label="Prediction", color="tomato", linestyle="--", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("y")
    ax.set_title(f"{title} (median run, median test series)\nrun test mean={run_mean:.4f}, series NMSE={med_nmse:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


prediction_plot(
    stored_models_a[median_run_a], stored_test_lists[median_run_a],
    run_results[median_run_a]["esn_feedback"]["test"]["scores"],
    run_means_a[median_run_a],
    "ESNFeedback — NARMA-30",
    os.path.join(HERE, "predictions_a.png"),
)
prediction_plot(
    stored_models_b[median_run_b], stored_test_lists[median_run_b],
    run_results[median_run_b]["esn_customizable"]["test"]["scores"],
    run_means_b[median_run_b],
    "ESNCustomizable — NARMA-30",
    os.path.join(HERE, "predictions_b.png"),
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
run_indices = list(range(N_RUNS))
ax1.bar(run_indices, run_means_a, color="steelblue")
ax1.axhline(run_means_a.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {run_means_a.mean():.4f}")
ax1.set_ylabel("Mean NMSE")
ax1.set_title("ESNFeedback — Mean test NMSE per run")
ax1.legend()
ax2.bar(run_indices, run_means_b, color="darkorange")
ax2.axhline(run_means_b.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {run_means_b.mean():.4f}")
ax2.set_xlabel("Run index")
ax2.set_ylabel("Mean NMSE")
ax2.set_title("ESNCustomizable — Mean test NMSE per run")
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "test_nmse_bar.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
x_min = min(run_means_a.min(), run_means_b.min()) - 0.01
x_max = max(run_means_a.max(), run_means_b.max()) + 0.01
x = np.linspace(x_min, x_max, 300)
for arr, label, color in [(run_means_a, "ESNFeedback", "steelblue"), (run_means_b, "ESNCustomizable", "darkorange")]:
    kde = gaussian_kde(arr, bw_method="scott")
    ax.plot(x, kde(x), label=f"{label} (mean={arr.mean():.4f})", color=color, linewidth=2)
    ax.fill_between(x, kde(x), alpha=0.15, color=color)
ax.set_xlabel("Mean NMSE")
ax.set_ylabel("Density")
ax.set_title(f"Distribution of run means — paired t-test p={p_value:.4f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "test_score_distribution.png"), dpi=150)
plt.close()

eigs = np.linalg.eigvals(stored_W_best[median_run_b])
theta = np.linspace(0, 2 * np.pi, 500)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.6)
ax.set_aspect("equal")
ax.set_title(f"ESNCustomizable — Eigenvalue distribution of W (run {median_run_b})")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.close()

print("Done.", flush=True)
