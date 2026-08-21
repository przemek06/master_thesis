import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sps

HERE = os.path.dirname(__file__)

WARMUP = 100
N_OPTUNA_TRIALS = 100
N_RESERVOIR = 400
ISO_ITERATIONS = 50
SUBSAMPLE = 6
HORIZON = 24
LAG = 12
N_INPUTS = 13 + LAG
N_OUTPUTS = 1
READOUT_INPUTS = True
RESERVOIR_SEED = 0
N_REPLICATES = 10
N_WORKERS = 5
SEEDS = list(range(N_REPLICATES))
MODELS = ["ginibre", "feedback"]

DPI = 150
WIDE = (7, 5)
PRIMARY = "darkorange"
SECONDARY = "steelblue"
REFERENCE = "black"


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def eval_nmse(model, series_list):
    scores = []
    for u, y in series_list:
        model.predict(u[:WARMUP])
        pred = model.predict(u[WARMUP:], initial_state=model.last_state)
        scores.append(nmse(pred, y[WARMUP:]))
    return float(np.mean(scores))


def persistence_nmse(series_list):
    scores = [nmse(y[WARMUP - HORIZON:-HORIZON], y[WARMUP:]) for _, y in series_list]
    return float(np.mean(scores))


def build_ginibre(p):
    from models.esn_customizable import ESNCustomizable
    from generate import generate_isospectral_sparse_matrix
    from distribution import sample_eigenvalues_ginibre
    np.random.seed(RESERVOIR_SEED)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=p["r_min"], r_max=p["r_max"], alpha=p["alpha"], size=size),
        N_RESERVOIR, p["sparsity"], iterations=ISO_ITERATIONS, seed=RESERVOIR_SEED,
    )
    return ESNCustomizable(
        n_inputs=N_INPUTS, n_reservoir=N_RESERVOIR, n_outputs=N_OUTPUTS,
        input_scaling=p["input_scaling"], leaky_rate=p["leaky_rate"],
        ridge=p["ridge"], noise=p["noise"], feedback_scaling=0.0,
        W=W, bias=np.array([0.2]), readout_inputs=READOUT_INPUTS, seed=RESERVOIR_SEED, device="cpu",
    )


def build_feedback(p):
    from models.esn_feedback import ESNFeedback
    return ESNFeedback(
        n_inputs=N_INPUTS, n_reservoir=N_RESERVOIR, n_outputs=N_OUTPUTS,
        spectral_radius=p["spectral_radius"], sparsity=p["sparsity"],
        input_scaling=p["input_scaling"], leaky_rate=p["leaky_rate"],
        ridge=p["ridge"], noise=p["noise"], feedback_scaling=0.0,
        readout_inputs=READOUT_INPUTS, seed=RESERVOIR_SEED, device="cpu",
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
    }


def feedback_params(trial):
    return {
        "spectral_radius": trial.suggest_float("spectral_radius", 0.1, 1.5),
        "sparsity": trial.suggest_float("sparsity", 0.9, 0.99),
        "input_scaling": trial.suggest_float("input_scaling", 0.1, 5.0),
        "leaky_rate": trial.suggest_float("leaky_rate", 0.01, 1.0, log=True),
        "ridge": trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
        "noise": trial.suggest_float("noise", 1e-6, 1e-2, log=True),
    }


BUILDERS = {"ginibre": (build_ginibre, ginibre_params), "feedback": (build_feedback, feedback_params)}


def run_one(task):
    model_name, opt_seed = task
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(RESERVOIR_SEED)
    import optuna
    from tasks.jena import load

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    build, sample_params = BUILDERS[model_name]
    u_train, y_train, val_list, test_list = load(horizon=HORIZON, lag=LAG, subsample=SUBSAMPLE)

    def objective(trial):
        model = build(sample_params(trial))
        model.fit(u_train, y_train, warmup=WARMUP)
        model.noise = 0.0
        return eval_nmse(model, val_list)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=opt_seed))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best = study.best_params
    model = build(best)
    model.fit(u_train, y_train, warmup=WARMUP)
    model.noise = 0.0
    val = eval_nmse(model, val_list)
    test = eval_nmse(model, test_list)

    print(f"  {model_name} opt_seed={opt_seed}: val {val:.6f} test {test:.6f}", flush=True)
    return {"model": model_name, "opt_seed": opt_seed, "val": val, "test": test, "best_params": best}


def stats(scores):
    arr = np.array(scores)
    return {
        "scores": arr.tolist(),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def main():
    tasks = [(m, s) for m in MODELS for s in SEEDS]
    print(f"Running {len(tasks)} tasks ({N_REPLICATES} optuna seeds x {len(MODELS)} models) on {N_WORKERS} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        outputs = list(executor.map(run_one, tasks))

    from tasks.jena import load
    _, _, _, test_list = load(horizon=HORIZON, lag=LAG, subsample=SUBSAMPLE)
    persistence = persistence_nmse(test_list)

    by_model = {m: [r for r in outputs if r["model"] == m] for m in MODELS}
    test_scores = {m: [r["test"] for r in by_model[m]] for m in MODELS}

    g, fb = np.array(test_scores["ginibre"]), np.array(test_scores["feedback"])
    t_stat, p_value = sps.ttest_ind(g, fb, equal_var=False)
    better = "ginibre" if g.mean() < fb.mean() else "feedback"

    results = {
        "config": {
            "task": "jena",
            "subsample": SUBSAMPLE,
            "sampling_interval_minutes": 10 * SUBSAMPLE,
            "horizon": HORIZON,
            "lag": LAG,
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_reservoir": N_RESERVOIR,
            "iso_iterations": ISO_ITERATIONS,
            "readout_inputs": READOUT_INPUTS,
            "warmup": WARMUP,
            "reservoir_seed": RESERVOIR_SEED,
            "optuna_seeds": SEEDS,
            "note": "hourly sampling; reservoir and data fixed, only the optuna (TPE) seed varies. "
                    "feedback disabled for both models; noise disabled at evaluation",
        },
        "persistence_test_nmse": persistence,
        "ginibre": stats(test_scores["ginibre"]),
        "feedback": stats(test_scores["feedback"]),
        "statistical_test": {
            "type": "welch_t_test",
            "note": "unpaired: both models see identical data in every replicate, so the samples differ only by search seed",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "better_model": better,
        },
        "runs": outputs,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("=== Optuna-seed variance (full retraining) | jena hourly ===\n")
        f.write(f"Sampling: every {SUBSAMPLE} rows = {10 * SUBSAMPLE} minutes\n")
        f.write(f"Horizon: {HORIZON} steps = {10 * SUBSAMPLE * HORIZON / 60:.0f} hours ahead\n")
        f.write(f"Optuna seeds: {SEEDS}\n")
        f.write(f"Persistence test NMSE: {persistence:.6f}\n\n")
        for m in MODELS:
            s = stats(test_scores[m])
            f.write(f"{m}:\n")
            f.write(f"  test mean +/- std: {s['mean']:.6f} +/- {s['std']:.6f}\n")
            f.write(f"  test median:       {s['median']:.6f}\n")
            f.write(f"  test min / max:    {s['min']:.6f} / {s['max']:.6f}\n\n")
        f.write(f"Welch t-test: t={t_stat:.3f}, p={p_value:.4f} "
                f"({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05)\n")
        f.write(f"Lower test NMSE: {better}\n")

    print(f"\nGinibre  test NMSE: {g.mean():.6f} +/- {g.std():.6f}")
    print(f"Feedback test NMSE: {fb.mean():.6f} +/- {fb.std():.6f}")
    print(f"Welch t={t_stat:.3f}, p={p_value:.4f}")

    fig, ax = plt.subplots(figsize=WIDE)
    data = [test_scores["ginibre"], test_scores["feedback"]]
    ax.boxplot(data, showmeans=True, widths=0.5)
    ax.set_xticklabels(["ginibre", "feedback"])
    for i, (scores, color) in enumerate(zip(data, [PRIMARY, SECONDARY])):
        x = np.random.default_rng(0).normal(i + 1, 0.05, len(scores))
        ax.scatter(x, scores, color=color, alpha=0.7, zorder=3, s=25)
    ax.axhline(persistence, color=REFERENCE, linestyle="--", linewidth=1,
               label=f"persistence = {persistence:.4f}")
    ax.plot([], [], " ", label=f"Welch p = {p_value:.4f}")
    ax.set_ylabel("test NMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "optuna_seed_variance.png"), dpi=DPI)
    plt.close(fig)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
