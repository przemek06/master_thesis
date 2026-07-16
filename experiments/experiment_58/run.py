import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import copy
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(__file__)
WARMUP = 200
HORIZON = 24
N_RUNS = 10
N_WORKERS = 5
N_TRAIN = 10
N_VAL = 10
N_TEST = 100
N_OPTUNA_TRIALS = 100
N_RESERVOIR = 400
N_INPUTS = 1
N_OUTPUTS = 1
SEED = 0
OPTUNA_SEED = 0
PLOT_START = 50
PLOT_SHOW = 24 * 14


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def eval_nmse(model, series_list):
    scores = []
    for u, y in series_list:
        model.predict(u[:WARMUP])
        pred = model.predict(u[WARMUP:], initial_state=model.last_state)
        scores.append(nmse(pred, y[WARMUP:]))
    return scores


def persistence_nmse(series_list):
    return [nmse(u[WARMUP:], y[WARMUP:]) for u, y in series_list]


def series_stats(scores):
    arr = np.array(scores)
    return {
        "scores": [float(s) for s in scores],
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
    }


def run_one(run_idx):
    import torch
    torch.set_num_threads(1)
    import optuna
    from tasks.electricity_batch import load
    from models.esn_customizable import ESNCustomizable
    from generate import generate_isospectral_sparse_matrix
    from distribution import sample_eigenvalues_ginibre

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = np.random.default_rng(SEED)
    W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
    W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))

    u_train, y_train, val_list, test_list = load(
        n_val=N_VAL, n_test=N_TEST, n_train=N_TRAIN, horizon=HORIZON, seed_offset=run_idx
    )

    holder = {"model": None, "score": np.inf, "W": None}

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
            leaky_rate=trial.suggest_float("leaky_rate", 0.01, 1.0, log=True),
            ridge=trial.suggest_float("ridge", 1e-6, 1e-1, log=True),
            noise=trial.suggest_float("noise", 1e-6, 1e-2, log=True),
            input_scaling=trial.suggest_float("input_scaling", 0.1, 5.0),
            feedback_scaling=0.0,
            W_in=W_in_fixed,
            W=W,
            W_fb=W_fb_fixed,
            bias=np.array([0.2]),
            seed=SEED,
            device="cpu",
        )
        model.fit_batch(u_train, y_train, warmup=WARMUP)
        score = float(np.mean(eval_nmse(model, val_list)))
        if score < holder["score"]:
            holder["score"] = score
            holder["model"] = copy.deepcopy(model)
            holder["W"] = W
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    model = holder["model"]
    model.reset_state()
    val_scores = eval_nmse(model, val_list)
    model.reset_state()
    test_scores = eval_nmse(model, test_list)
    baseline_scores = persistence_nmse(test_list)

    median_idx = int(np.argsort(np.array(test_scores))[len(test_scores) // 2])
    u_med, y_med = test_list[median_idx]
    model.predict(u_med[:WARMUP])
    pred_med = model.predict(u_med[WARMUP:], initial_state=model.last_state)

    result = {
        "run_index": run_idx,
        "best_val_score_optuna": float(study.best_value),
        "val": series_stats(val_scores),
        "test": series_stats(test_scores),
        "persistence": series_stats(baseline_scores),
        "best_params": {k: float(v) for k, v in study.best_params.items()},
    }
    plot = {
        "target": np.asarray(y_med[WARMUP:, 0], dtype=np.float64),
        "pred": np.asarray(pred_med[:, 0], dtype=np.float64),
        "series_nmse": float(test_scores[median_idx]),
        "eigenvalues": np.linalg.eigvals(holder["W"]),
    }
    print(f"  Run {run_idx} done: test mean {result['test']['mean']:.4f} "
          f"median {result['test']['median']:.4f}, persistence mean {result['persistence']['mean']:.4f}",
          flush=True)
    return result, plot


def main():
    print(f"Running {N_RUNS} runs on {N_WORKERS} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        outputs = list(executor.map(run_one, range(N_RUNS)))

    run_results = [o[0] for o in outputs]
    plots = [o[1] for o in outputs]

    run_means = np.array([r["test"]["mean"] for r in run_results])
    run_medians = np.array([r["test"]["median"] for r in run_results])
    base_means = np.array([r["persistence"]["mean"] for r in run_results])

    results = {
        "config": {
            "model": "ESNCustomizable (feedback_scaling=0)",
            "task": "electricity_batch",
            "horizon": HORIZON,
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
            "test_run_means": run_means.tolist(),
            "mean_of_means": float(run_means.mean()),
            "std_of_means": float(run_means.std()),
            "median_of_medians": float(np.median(run_medians)),
            "persistence_mean_of_means": float(base_means.mean()),
            "beats_persistence": bool(run_means.mean() < base_means.mean()),
        },
    }

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("=== Summary ===\n")
        f.write(f"ESNCustomizable test mean of means: {run_means.mean():.4f} +/- {run_means.std():.4f}\n")
        f.write(f"ESNCustomizable test median of medians: {np.median(run_medians):.4f}\n")
        f.write(f"Persistence     test mean of means: {base_means.mean():.4f}\n")
        f.write(f"Beats persistence: {run_means.mean() < base_means.mean()}\n")
        f.write("\n=== Per-run results ===\n")
        for r in run_results:
            f.write(f"\nRun {r['run_index']}\n")
            f.write(f"  val mean: {r['val']['mean']:.4f}, test mean: {r['test']['mean']:.4f}, "
                    f"test median: {r['test']['median']:.4f}, persistence mean: {r['persistence']['mean']:.4f}\n")
            f.write(f"  best params: {r['best_params']}\n")

    print("Plotting...", flush=True)

    median_run = int(np.argsort(run_means)[N_RUNS // 2])
    plot = plots[median_run]
    target = plot["target"]
    pred = plot["pred"]

    t = np.arange(WARMUP + PLOT_START, WARMUP + PLOT_START + PLOT_SHOW)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, target[PLOT_START:PLOT_START + PLOT_SHOW], label="Target", color="steelblue")
    ax.plot(t, pred[PLOT_START:PLOT_START + PLOT_SHOW], label="Prediction", color="tomato", linestyle="--", alpha=0.8)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Standardized load")
    ax.set_title(f"ESNCustomizable — Electricity {HORIZON}h ahead (median run, median test client)\n"
                 f"run test mean={run_means[median_run]:.4f}, client NMSE={plot['series_nmse']:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(N_RUNS), run_means, color="darkorange", label="ESNCustomizable")
    ax.plot(range(N_RUNS), base_means, color="black", marker="o", linestyle="--", linewidth=1.5, label="Persistence")
    ax.axhline(run_means.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {run_means.mean():.4f}")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Mean test NMSE")
    ax.set_title("ESNCustomizable — Mean test NMSE per run")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "test_nmse_bar.png"), dpi=150)
    plt.close()

    all_scores = np.concatenate([np.array(r["test"]["scores"]) for r in run_results])
    all_base = np.concatenate([np.array(r["persistence"]["scores"]) for r in run_results])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(0, np.percentile(np.concatenate([all_scores, all_base]), 99), 300)
    for arr, label, color in [(all_scores, "ESNCustomizable", "darkorange"), (all_base, "Persistence", "black")]:
        kde = gaussian_kde(arr, bw_method="scott")
        ax.plot(x, kde(x), label=f"{label} (median={np.median(arr):.4f})", color=color, linewidth=2)
        ax.fill_between(x, kde(x), alpha=0.15, color=color)
    ax.set_xlabel("Test NMSE per client")
    ax.set_ylabel("Density")
    ax.set_title(f"Per-client test NMSE across {N_RUNS} runs x {N_TEST} held-out clients")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "test_score_distribution.png"), dpi=150)
    plt.close()

    eigs = plots[median_run]["eigenvalues"]
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


if __name__ == "__main__":
    main()
