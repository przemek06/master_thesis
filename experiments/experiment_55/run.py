import sys
import os
print("Setting up environment...", flush=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import optuna
from tasks.memory_capacity import load, K_MAX
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
N_RESERVOIR = 100
N_INPUTS = 1
N_OUTPUTS = K_MAX
SEED = 0
OPTUNA_SEED = 0


def eval_mc(model, series_list):
    totals, curves = [], []
    for u, y in series_list:
        pred = model.predict(u)[WARMUP:]
        target = y[WARMUP:]
        mc_k = np.array([np.corrcoef(pred[:, k], target[:, k])[0, 1] ** 2 for k in range(K_MAX)])
        curves.append(mc_k)
        totals.append(float(mc_k.sum()))
    return totals, curves


def series_stats(scores):
    arr = np.array(scores)
    return {"scores": [float(s) for s in scores], "mean": float(arr.mean()), "std": float(arr.std())}


rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))

optuna.logging.set_verbosity(optuna.logging.WARNING)

run_results = []
run_test_curves = []
stored_W_best = []

for run_idx in range(N_RUNS):
    data_offset = run_idx * (N_TRAIN + N_VAL + N_TEST)
    print(f"\n=== Run {run_idx + 1}/{N_RUNS} (data_offset={data_offset}) ===", flush=True)

    u_train, y_train, val_list, test_list = load(n_val=N_VAL, n_test=N_TEST, n_train=N_TRAIN, seed_offset=data_offset)

    print("  Optimizing ESNCustomizable...", flush=True)

    holder = {"model": None, "score": -np.inf, "W": None}

    def objective(trial):
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
            feedback_scaling=0.0,
            W_in=W_in_fixed,
            W=W,
            W_fb=W_fb_fixed,
            bias=np.array([0.2]),
            seed=SEED,
        )
        model.fit_batch(u_train, y_train, warmup=WARMUP)
        totals, _ = eval_mc(model, val_list)
        score = float(np.mean(totals))
        if score > holder["score"]:
            holder["score"] = score
            holder["model"] = copy.deepcopy(model)
            holder["W"] = W
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)
    best_params = study.best_params

    model = holder["model"]
    val_totals, _ = eval_mc(model, val_list)
    test_totals, test_curves = eval_mc(model, test_list)
    mean_test_curve = np.mean(test_curves, axis=0)
    print(f"  ESNCustomizable val MC: {np.mean(val_totals):.2f}, test MC: {np.mean(test_totals):.2f}", flush=True)

    run_results.append({
        "run_index": run_idx,
        "data_offset": data_offset,
        "best_val_score_optuna": float(study.best_value),
        "val": series_stats(val_totals),
        "test": series_stats(test_totals),
        "mean_test_forgetting_curve": mean_test_curve.tolist(),
        "best_params": {k: float(v) for k, v in best_params.items()},
    })
    run_test_curves.append(mean_test_curve)
    stored_W_best.append(holder["W"])

run_means = np.array([r["test"]["mean"] for r in run_results])

results = {
    "config": {
        "model": "ESNCustomizable",
        "task": "memory_capacity",
        "k_max": K_MAX,
        "n_runs": N_RUNS,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_TEST,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "n_reservoir": N_RESERVOIR,
        "warmup": WARMUP,
        "feedback_scaling": 0.0,
        "seed": SEED,
        "optuna_seed": OPTUNA_SEED,
    },
    "runs": run_results,
    "summary": {
        "run_means": run_means.tolist(),
        "mean_of_means": float(run_means.mean()),
        "std_of_means": float(run_means.std()),
    },
}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write("=== Summary (memory capacity, higher is better) ===\n")
    f.write(f"ESNCustomizable test MC mean of means: {run_means.mean():.2f} ± {run_means.std():.2f} (bound: {N_RESERVOIR})\n")
    f.write("\n=== Per-run results ===\n")
    for r in run_results:
        f.write(f"\nRun {r['run_index']} (data_offset={r['data_offset']})\n")
        f.write(f"  val MC: {r['val']['mean']:.2f}, test MC: {r['test']['mean']:.2f}\n")

print("Plotting...", flush=True)

delays = np.arange(1, K_MAX + 1)
fig, ax = plt.subplots(figsize=(10, 5))
for curve in run_test_curves:
    ax.plot(delays, curve, color="darkorange", alpha=0.25, linewidth=1)
ax.plot(delays, np.mean(run_test_curves, axis=0), color="darkorange", linewidth=2.5, label="Mean over runs")
ax.set_xlabel("Delay k")
ax.set_ylabel("MC_k")
ax.set_title(f"ESNCustomizable — Forgetting curve (test), total MC = {run_means.mean():.2f} ± {run_means.std():.2f}")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "forgetting_curve.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(N_RUNS), run_means, color="darkorange")
ax.axhline(run_means.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {run_means.mean():.2f}")
ax.set_xlabel("Run index")
ax.set_ylabel("Mean test MC")
ax.set_title("ESNCustomizable — Mean test memory capacity per run")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "mc_bar.png"), dpi=150)
plt.close()

u, y = test_list[0]
pred = model.predict(u)[WARMUP:]
target = y[WARMUP:]
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, k in zip(axes.flat, [1, 5, 10, 25, 50, 100]):
    mc_k = np.corrcoef(pred[:, k - 1], target[:, k - 1])[0, 1] ** 2
    ax.scatter(target[:, k - 1], pred[:, k - 1], s=4, alpha=0.3, color="darkorange")
    ax.plot([-0.8, 0.8], [-0.8, 0.8], color="black", linewidth=1)
    ax.set_title(f"k={k}, MC_k={mc_k:.3f}")
    ax.set_xlabel("target u(t-k)")
    ax.set_ylabel("predicted")
fig.suptitle("ESNCustomizable — Target vs predicted (last run, first test series)")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "target_vs_pred.png"), dpi=150)
plt.close()

median_run = int(np.argsort(run_means)[N_RUNS // 2])
eigs = np.linalg.eigvals(stored_W_best[median_run])
theta = np.linspace(0, 2 * np.pi, 500)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.6)
ax.set_aspect("equal")
ax.set_title(f"ESNCustomizable — Eigenvalue distribution of W (run {median_run})")
ax.set_xlabel("Re")
ax.set_ylabel("Im")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
plt.close()

print("Done.", flush=True)
