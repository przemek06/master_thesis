import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(__file__)

WARMUP = 1000
N_OPTUNA_TRIALS = 100
N_RESERVOIR = 400
ISO_ITERATIONS = 50
N_INPUTS = 3
N_OUTPUTS = 3
READOUT_INPUTS = True
N_RUNS = 10
N_WORKERS = 5
N_TRAIN = 5
N_VAL = 10
N_TEST = 20
SERIES_LENGTH = 6000
SUBSAMPLE = 5
MAX_ROLLOUT = 2000
CHUNK = 250
HORIZON = 100
DIVERGENCE_WINDOW = 20
DIVERGENCE_THRESHOLD = 1.0
LYAPUNOV_STEPS = 22
INPUT_SEED = 0
MODELS = ["ginibre", "feedback"]

_rng = np.random.default_rng(INPUT_SEED)
W_IN = _rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_FB = _rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def steps_until_divergence(pred, target):
    for i in range(DIVERGENCE_WINDOW, len(pred) + 1):
        if nmse(pred[i - DIVERGENCE_WINDOW:i], target[i - DIVERGENCE_WINDOW:i]) > DIVERGENCE_THRESHOLD:
            return i - DIVERGENCE_WINDOW
    return len(pred)


def eval_series(model, u, y):
    model.reset_state()
    model.predict(u[:WARMUP])
    target = y[WARMUP:WARMUP + MAX_ROLLOUT]
    chunks = []
    n = 0
    while n < MAX_ROLLOUT:
        chunks.append(model.predict_autonomous(min(CHUNK, MAX_ROLLOUT - n)))
        pred = np.concatenate(chunks)
        n = len(pred)
        steps = steps_until_divergence(pred, target[:n])
        if steps < n:
            break
    return steps, float(nmse(pred[:HORIZON], target[:HORIZON])), pred, target


def eval_all(model, series_list):
    out = [eval_series(model, u, y) for u, y in series_list]
    return [o[0] for o in out], [o[1] for o in out]


def build_ginibre(p, seed):
    from models.esn_customizable import ESNCustomizable
    from generate import generate_isospectral_sparse_matrix
    from distribution import sample_eigenvalues_ginibre
    np.random.seed(seed)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=p["r_min"], r_max=p["r_max"], alpha=p["alpha"], size=size),
        N_RESERVOIR, p["sparsity"], iterations=ISO_ITERATIONS, seed=seed,
    )
    return ESNCustomizable(
        n_inputs=N_INPUTS, n_reservoir=N_RESERVOIR, n_outputs=N_OUTPUTS,
        input_scaling=p["input_scaling"], leaky_rate=p["leaky_rate"],
        ridge=p["ridge"], noise=p["noise"], feedback_scaling=p["feedback_scaling"],
        W=W, W_in=W_IN, W_fb=W_FB, bias=np.array([0.2]),
        readout_inputs=READOUT_INPUTS, seed=seed, device="cpu",
    )


def build_feedback(p, seed):
    from models.esn_feedback import ESNFeedback
    return ESNFeedback(
        n_inputs=N_INPUTS, n_reservoir=N_RESERVOIR, n_outputs=N_OUTPUTS,
        spectral_radius=p["spectral_radius"], sparsity=p["sparsity"],
        input_scaling=p["input_scaling"], leaky_rate=p["leaky_rate"],
        ridge=p["ridge"], noise=p["noise"], feedback_scaling=p["feedback_scaling"],
        W_in=W_IN, W_fb=W_FB,
        readout_inputs=READOUT_INPUTS, seed=seed, device="cpu",
    )


def ginibre_params(trial):
    r_min = trial.suggest_float("r_min", 0.0, 0.95)
    return {
        "r_min": r_min,
        "r_max": trial.suggest_float("r_max", r_min, 1.0),
        "alpha": trial.suggest_float("alpha", 0.1, 10.0),
        "sparsity": trial.suggest_float("sparsity", 0.9, 0.99),
        "input_scaling": trial.suggest_float("input_scaling", 0.1, 5.0),
        "leaky_rate": trial.suggest_float("leaky_rate", 0.01, 1.0, log=True),
        "ridge": trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
        "noise": trial.suggest_float("noise", 1e-6, 1e-2, log=True),
        "feedback_scaling": trial.suggest_float("feedback_scaling", 0.0, 1.0),
    }


def feedback_params(trial):
    return {
        "spectral_radius": trial.suggest_float("spectral_radius", 0.1, 1.5),
        "sparsity": trial.suggest_float("sparsity", 0.9, 0.99),
        "input_scaling": trial.suggest_float("input_scaling", 0.1, 5.0),
        "leaky_rate": trial.suggest_float("leaky_rate", 0.01, 1.0, log=True),
        "ridge": trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
        "noise": trial.suggest_float("noise", 1e-6, 1e-2, log=True),
        "feedback_scaling": trial.suggest_float("feedback_scaling", 0.0, 1.0),
    }


BUILDERS = {"ginibre": (build_ginibre, ginibre_params), "feedback": (build_feedback, feedback_params)}


def stats_of(scores):
    arr = np.array(scores, dtype=float)
    return {
        "scores": arr.tolist(),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_one(task):
    model_name, run_idx = task
    import torch
    torch.set_num_threads(1)
    import optuna
    from tasks.lorenz_batch import load

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    build, sample_params = BUILDERS[model_name]
    seed_offset = run_idx * (N_TRAIN + N_VAL + N_TEST)
    u_train, y_train, val_list, test_list = load(
        n_val=N_VAL, n_test=N_TEST, n_train=N_TRAIN,
        series_length=SERIES_LENGTH, subsample=SUBSAMPLE, seed_offset=seed_offset,
    )

    def objective(trial):
        model = build(sample_params(trial), run_idx)
        model.fit_batch(u_train, y_train, warmup=WARMUP)
        model.noise = 0.0
        steps, _ = eval_all(model, val_list)
        return float(np.mean(steps))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=run_idx))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best = study.best_params
    model = build(best, run_idx)
    model.fit_batch(u_train, y_train, warmup=WARMUP)
    model.noise = 0.0
    val_steps, val_horizon = eval_all(model, val_list)
    test_steps, test_horizon = eval_all(model, test_list)

    median_idx = int(np.argsort(np.array(test_steps))[len(test_steps) // 2])
    u_med, y_med = test_list[median_idx]
    _, _, pred_med, target_med = eval_series(model, u_med, y_med)

    result = {
        "model": model_name,
        "run_index": run_idx,
        "seed_offset": seed_offset,
        "best_val_score_optuna": float(study.best_value),
        "val_steps": stats_of(val_steps),
        "test_steps": stats_of(test_steps),
        "val_horizon_nmse": stats_of(val_horizon),
        "test_horizon_nmse": stats_of(test_horizon),
        "best_params": {k: float(v) for k, v in best.items()},
    }
    plot = {
        "pred": pred_med[:, 0].astype(np.float64),
        "target": target_med[:len(pred_med), 0].astype(np.float64),
        "steps": int(test_steps[median_idx]),
        "eigenvalues": np.linalg.eigvals(model.W.cpu().numpy()) if model_name == "ginibre" else None,
    }
    print(f"  {model_name} run {run_idx}: test steps {result['test_steps']['mean']:.1f}, "
          f"horizon NMSE {result['test_horizon_nmse']['median']:.4f}", flush=True)
    return result, plot


def main():
    tasks = [(m, r) for m in MODELS for r in range(N_RUNS)]
    print(f"Running {len(tasks)} tasks ({N_RUNS} runs x {len(MODELS)} models) on {N_WORKERS} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        outputs = list(executor.map(run_one, tasks))

    run_results = [o[0] for o in outputs]
    plots = {(o[0]["model"], o[0]["run_index"]): o[1] for o in outputs}

    by_model = {m: sorted([r for r in run_results if r["model"] == m], key=lambda r: r["run_index"]) for m in MODELS}
    steps_means = {m: np.array([r["test_steps"]["mean"] for r in by_model[m]]) for m in MODELS}
    horizon_scores = {m: np.array([r["test_horizon_nmse"]["median"] for r in by_model[m]]) for m in MODELS}

    t_steps, p_steps = stats.ttest_rel(steps_means["ginibre"], steps_means["feedback"])
    t_horizon, p_horizon = stats.ttest_rel(horizon_scores["ginibre"], horizon_scores["feedback"])

    results = {
        "config": {
            "task": "lorenz",
            "n_runs": N_RUNS,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "series_length": SERIES_LENGTH,
            "subsample": SUBSAMPLE,
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_reservoir": N_RESERVOIR,
            "iso_iterations": ISO_ITERATIONS,
            "readout_inputs": READOUT_INPUTS,
            "warmup": WARMUP,
            "max_rollout": MAX_ROLLOUT,
            "horizon": HORIZON,
            "divergence_window": DIVERGENCE_WINDOW,
            "divergence_threshold": DIVERGENCE_THRESHOLD,
            "lyapunov_steps": LYAPUNOV_STEPS,
            "input_seed": INPUT_SEED,
            "note": "data, reservoir seed and optuna seed all vary per run; noise disabled at evaluation",
        },
        "runs": run_results,
        "summary": {
            m: {
                "steps_run_means": steps_means[m].tolist(),
                "steps_mean_of_means": float(steps_means[m].mean()),
                "steps_std_of_means": float(steps_means[m].std()),
                "steps_lyapunov_times": float(steps_means[m].mean() / LYAPUNOV_STEPS),
                "horizon_nmse_run_scores": horizon_scores[m].tolist(),
                "horizon_nmse_mean": float(horizon_scores[m].mean()),
                "horizon_nmse_std": float(horizon_scores[m].std()),
            }
            for m in MODELS
        },
        "statistical_test": {
            "steps": {"t_statistic": float(t_steps), "p_value": float(p_steps), "significant": bool(p_steps < 0.05)},
            "horizon_nmse": {"t_statistic": float(t_horizon), "p_value": float(p_horizon), "significant": bool(p_horizon < 0.05)},
            "censored_runs": int(sum(s >= MAX_ROLLOUT for m in MODELS for r in by_model[m] for s in r["test_steps"]["scores"])),
        },
    }

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("=== Lorenz, 10 runs with per-run data ===\n")
        for m in MODELS:
            s = results["summary"][m]
            f.write(f"{m}:\n")
            f.write(f"  steps until divergence: {s['steps_mean_of_means']:.1f} +/- {s['steps_std_of_means']:.1f} "
                    f"({s['steps_lyapunov_times']:.2f} Lyapunov times)\n")
            f.write(f"  horizon-{HORIZON} NMSE:   {s['horizon_nmse_mean']:.5f} +/- {s['horizon_nmse_std']:.5f}\n")
        f.write(f"\nPaired t-test (steps):        t={t_steps:.3f}, p={p_steps:.4f}\n")
        f.write(f"Paired t-test (horizon NMSE): t={t_horizon:.3f}, p={p_horizon:.4f}\n")
        f.write(f"Censored test rollouts (hit {MAX_ROLLOUT}): {results['statistical_test']['censored_runs']}\n")
        f.write("\n=== Per-run results ===\n")
        for r in range(N_RUNS):
            f.write(f"\nRun {r}\n")
            for m in MODELS:
                res = by_model[m][r]
                f.write(f"  {m:9s} val steps: {res['val_steps']['mean']:7.1f}, test steps: {res['test_steps']['mean']:7.1f}, "
                        f"test horizon NMSE: {res['test_horizon_nmse']['median']:.5f}\n")

    print("Plotting...", flush=True)
    colors = {"ginibre": "darkorange", "feedback": "steelblue"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    width = 0.38
    idx = np.arange(N_RUNS)
    for i, m in enumerate(MODELS):
        ax1.bar(idx + (i - 0.5) * width, steps_means[m], width, label=m, color=colors[m])
        ax2.bar(idx + (i - 0.5) * width, horizon_scores[m], width, label=m, color=colors[m])
    ax1.set_xlabel("Run")
    ax1.set_ylabel("Mean steps until divergence")
    ax1.set_title("Steps until divergence per run")
    ax1.legend()
    ax2.set_xlabel("Run")
    ax2.set_ylabel(f"Median horizon-{HORIZON} NMSE")
    ax2.set_title(f"Horizon-{HORIZON} NMSE per run")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "per_run_scores.png"), dpi=150)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    ax1.boxplot([steps_means[m] for m in MODELS], showmeans=True, widths=0.5)
    ax1.set_xticklabels(MODELS)
    ax2.boxplot([horizon_scores[m] for m in MODELS], showmeans=True, widths=0.5)
    ax2.set_xticklabels(MODELS)
    for i, m in enumerate(MODELS):
        jitter = np.random.default_rng(0).normal(i + 1, 0.04, N_RUNS)
        ax1.scatter(jitter, steps_means[m], color=colors[m], alpha=0.7, zorder=3, s=25)
        ax2.scatter(jitter, horizon_scores[m], color=colors[m], alpha=0.7, zorder=3, s=25)
    ax1.set_ylabel("Mean steps until divergence")
    ax1.set_title(f"Run means, paired t-test p={p_steps:.4f}")
    ax2.set_ylabel(f"Median horizon-{HORIZON} NMSE")
    ax2.set_title(f"Run means, paired t-test p={p_horizon:.4f}")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "run_mean_distribution.png"), dpi=150)
    plt.close()

    fig, axes = plt.subplots(len(MODELS), 1, figsize=(14, 7), sharex=True)
    for ax, m in zip(axes, MODELS):
        median_run = int(np.argsort(steps_means[m])[N_RUNS // 2])
        p = plots[(m, median_run)]
        span = min(len(p["pred"]), 2 * max(p["steps"], HORIZON))
        ax.plot(p["target"][:span], label="Target", color="black", linewidth=1)
        ax.plot(p["pred"][:span], label="Prediction", color=colors[m], linestyle="--")
        ax.axvline(p["steps"], color="red", linestyle="--", linewidth=1, label=f"Divergence at {p['steps']}")
        ax.set_ylabel("x")
        ax.set_title(f"{m} — median run {median_run}, median test series")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Autonomous step")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
    plt.close()

    median_run = int(np.argsort(steps_means["ginibre"])[N_RUNS // 2])
    eigs = plots[("ginibre", median_run)]["eigenvalues"]
    theta = np.linspace(0, 2 * np.pi, 500)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
    ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.6, color=colors["ginibre"])
    ax.set_aspect("equal")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(f"Ginibre eigenvalues, median run {median_run}")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
    plt.close()

    print(f"\nginibre  steps {steps_means['ginibre'].mean():.1f} +/- {steps_means['ginibre'].std():.1f}")
    print(f"feedback steps {steps_means['feedback'].mean():.1f} +/- {steps_means['feedback'].std():.1f}")
    print(f"paired t-test p={p_steps:.4f}")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
