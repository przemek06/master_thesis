import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

WARMUP = 100
N_OPTUNA_TRIALS = 100
N_RESERVOIR = 400
ISO_ITERATIONS = 50
RESERVOIR_SEED = 0
N_REPLICATES = 10
N_WORKERS = 5
SEEDS = list(range(N_REPLICATES))
MODELS = ["esn", "ginibre"]


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def eval_nmse(model, u, y):
    model.predict(u[:WARMUP])
    pred = model.predict(u[WARMUP:], initial_state=model.last_state)
    return nmse(pred, y[WARMUP:])


def build_esn(p):
    from models.esn import ESN
    return ESN(
        n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1,
        spectral_radius=p["spectral_radius"], sparsity=p["sparsity"],
        input_scaling=p["input_scaling"], leaky_rate=p["leaky_rate"],
        ridge=p["ridge"], noise=p["noise"], seed=RESERVOIR_SEED, device="cpu",
    )


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
        n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1,
        input_scaling=p["input_scaling"], leaky_rate=p["leaky_rate"],
        ridge=p["ridge"], noise=p["noise"], feedback_scaling=0.0,
        W=W, bias=np.array([0.2]), seed=RESERVOIR_SEED, device="cpu",
    )


def esn_params(trial):
    return {
        "spectral_radius": trial.suggest_float("spectral_radius", 0.1, 1.5),
        "sparsity": trial.suggest_float("sparsity", 0.9, 0.99),
        "input_scaling": trial.suggest_float("input_scaling", 0.1, 5.0),
        "leaky_rate": trial.suggest_float("leaky_rate", 0.01, 1.0, log=True),
        "ridge": trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
        "noise": trial.suggest_float("noise", 1e-6, 1e-2, log=True),
    }


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


BUILDERS = {"esn": (build_esn, esn_params), "ginibre": (build_ginibre, ginibre_params)}


def run_one(task):
    model_name, opt_seed = task
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(RESERVOIR_SEED)
    import optuna
    from tasks.santafe_laser import load

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    build, sample_params = BUILDERS[model_name]
    u_train, y_train, u_val, y_val, u_test, y_test = load()

    def objective(trial):
        model = build(sample_params(trial))
        model.fit(u_train, y_train, warmup=WARMUP)
        return float(eval_nmse(model, u_val, y_val))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=opt_seed))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best = study.best_params
    model = build(best)
    model.fit(u_train, y_train, warmup=WARMUP)
    val = float(eval_nmse(model, u_val, y_val))
    test = float(eval_nmse(model, u_test, y_test))

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

    from tasks.santafe_laser import load
    _, _, _, _, u_test, y_test = load()
    persistence = float(nmse(u_test[WARMUP:], y_test[WARMUP:]))

    by_model = {m: [r for r in outputs if r["model"] == m] for m in MODELS}
    test_scores = {m: [r["test"] for r in by_model[m]] for m in MODELS}

    results = {
        "config": {
            "task": "santafe_laser",
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_reservoir": N_RESERVOIR,
            "iso_iterations": ISO_ITERATIONS,
            "warmup": WARMUP,
            "reservoir_seed": RESERVOIR_SEED,
            "optuna_seeds": SEEDS,
            "note": "reservoir fixed; only the optuna (TPE) seed varies -> variance of the full tuning process",
        },
        "persistence_test_nmse": persistence,
        "esn": stats(test_scores["esn"]),
        "ginibre": stats(test_scores["ginibre"]),
        "runs": outputs,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("=== Optuna-seed variance (full retraining) | santafe_laser ===\n")
        f.write(f"Optuna seeds: {SEEDS}\n")
        f.write(f"Persistence test NMSE: {persistence:.6f}\n\n")
        for m in MODELS:
            s = stats(test_scores[m])
            f.write(f"{m}:\n")
            f.write(f"  test mean +/- std: {s['mean']:.6f} +/- {s['std']:.6f}\n")
            f.write(f"  test median:       {s['median']:.6f}\n")
            f.write(f"  test min / max:    {s['min']:.6f} / {s['max']:.6f}\n\n")

    print(f"\nESN     test NMSE: {np.mean(test_scores['esn']):.6f} +/- {np.std(test_scores['esn']):.6f}")
    print(f"Ginibre test NMSE: {np.mean(test_scores['ginibre']):.6f} +/- {np.std(test_scores['ginibre']):.6f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [test_scores["esn"], test_scores["ginibre"]]
    colors = ["steelblue", "darkorange"]
    ax.boxplot(data, labels=["ESN", "Ginibre"], showmeans=True, widths=0.5)
    for i, (scores, color) in enumerate(zip(data, colors)):
        x = np.random.default_rng(0).normal(i + 1, 0.05, len(scores))
        ax.scatter(x, scores, color=color, alpha=0.7, zorder=3, s=25)
    ax.set_ylabel("Test NMSE")
    ax.set_title(f"Test NMSE across {N_REPLICATES} optuna seeds (full retraining, fixed split & reservoir)")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "optuna_seed_variance.png"), dpi=150)
    plt.close()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
