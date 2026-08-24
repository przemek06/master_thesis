import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
WARMUP = 100
N_RUNS = 10
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
N_WORKERS = 5
MODELS = ["feedback", "customizable"]


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def evaluate(model, series_list):
    scores = []
    preds = []
    for u, y in series_list:
        model.predict(u[:WARMUP])
        pred = model.predict(u[WARMUP:], initial_state=model.last_state)
        scores.append(nmse(pred, y[WARMUP:]))
        preds.append(np.asarray(pred))
    return float(np.mean(scores)), preds


def persistence_nmse(series_list):
    scores = [nmse(y[WARMUP - HORIZON:-HORIZON], y[WARMUP:]) for _, y in series_list]
    return float(np.mean(scores))


def build_customizable(p):
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


def customizable_params(trial):
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


BUILDERS = {
    "customizable": (build_customizable, customizable_params),
    "feedback": (build_feedback, feedback_params),
}


def run_one(task):
    model_name, run_idx = task
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(RESERVOIR_SEED)
    import optuna
    from tasks.jena import load

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    build, sample_params = BUILDERS[model_name]
    u_train, y_train, val_list, test_list = load(horizon=HORIZON, lag=LAG, subsample=SUBSAMPLE)
    optuna_seed = run_idx

    def objective(trial):
        model = build(sample_params(trial))
        model.fit(u_train, y_train, warmup=WARMUP)
        model.noise = 0.0
        return evaluate(model, val_list)[0]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=optuna_seed))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best = study.best_params
    model = build(best)
    model.fit(u_train, y_train, warmup=WARMUP)
    model.noise = 0.0
    val, _ = evaluate(model, val_list)
    test, test_preds = evaluate(model, test_list)

    print(f"  {model_name} run {run_idx}: val {val:.6f} test {test:.6f}", flush=True)
    return {
        "model": model_name,
        "run_index": run_idx,
        "optuna_seed": optuna_seed,
        "val": val,
        "test": test,
        "best_params": best,
        "W": model.W.numpy(),
        "test_pred": test_preds[0][:, 0],
    }


def main():
    tasks = [(m, r) for m in MODELS for r in range(N_RUNS)]
    print(f"Running {len(tasks)} tasks ({N_RUNS} runs x {len(MODELS)} models) on {N_WORKERS} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        outputs = list(executor.map(run_one, tasks))

    by_model = {m: sorted([o for o in outputs if o["model"] == m], key=lambda o: o["run_index"]) for m in MODELS}
    run_means = {m: np.array([o["test"] for o in by_model[m]]) for m in MODELS}
    median_run = {m: int(np.argsort(run_means[m])[N_RUNS // 2]) for m in MODELS}

    from tasks.jena import load
    _, _, _, test_list = load(horizon=HORIZON, lag=LAG, subsample=SUBSAMPLE)
    persistence = persistence_nmse(test_list)
    test_target = test_list[0][1][WARMUP:, 0]

    t_stat, p_value = stats.ttest_ind(run_means["feedback"], run_means["customizable"], equal_var=False)
    better = "customizable" if run_means["customizable"].mean() < run_means["feedback"].mean() else "feedback"

    W_stacks = {m: np.stack([o["W"] for o in by_model[m]]) for m in MODELS}
    np.savez_compressed(
        os.path.join(HERE, "arrays.npz"),
        W_feedback=W_stacks["feedback"],
        W_customizable=W_stacks["customizable"],
        run_means_feedback=run_means["feedback"],
        run_means_customizable=run_means["customizable"],
        median_run_feedback=median_run["feedback"],
        median_run_customizable=median_run["customizable"],
        test_target=test_target,
        pred_feedback=by_model["feedback"][median_run["feedback"]]["test_pred"],
        pred_customizable=by_model["customizable"][median_run["customizable"]]["test_pred"],
    )

    run_results = []
    for r in range(N_RUNS):
        fb = by_model["feedback"][r]
        cu = by_model["customizable"][r]
        run_results.append({
            "run_index": r,
            "optuna_seed": fb["optuna_seed"],
            "feedback": {k: fb[k] for k in ("val", "test", "best_params")},
            "customizable": {k: cu[k] for k in ("val", "test", "best_params")},
        })

    results = {
        "config": {
            "task": "jena",
            "subsample": SUBSAMPLE,
            "sampling_interval_minutes": 10 * SUBSAMPLE,
            "horizon": HORIZON,
            "horizon_hours": 10 * SUBSAMPLE * HORIZON / 60,
            "lag": LAG,
            "n_runs": N_RUNS,
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_reservoir": N_RESERVOIR,
            "iso_iterations": ISO_ITERATIONS,
            "readout_inputs": READOUT_INPUTS,
            "warmup": WARMUP,
            "reservoir_seed": RESERVOIR_SEED,
            "n_workers": N_WORKERS,
            "note": "hourly sampling, optuna-seed variance over runs; feedback disabled for both, noise disabled at evaluation",
        },
        "persistence_test_nmse": persistence,
        "runs": run_results,
        "summary": {
            "feedback": {
                "run_means": run_means["feedback"].tolist(),
                "mean_of_means": float(run_means["feedback"].mean()),
                "std_of_means": float(run_means["feedback"].std()),
            },
            "customizable": {
                "run_means": run_means["customizable"].tolist(),
                "mean_of_means": float(run_means["customizable"].mean()),
                "std_of_means": float(run_means["customizable"].std()),
            },
            "statistical_test": {
                "type": "welch_t_test",
                "note": "unpaired: both models see identical data in every run, so the samples differ only by search seed",
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
        f.write("=== Optuna-seed variance | jena hourly ===\n")
        f.write(f"Sampling: every {SUBSAMPLE} rows = {10 * SUBSAMPLE} minutes\n")
        f.write(f"Horizon: {HORIZON} steps = {10 * SUBSAMPLE * HORIZON / 60:.0f} hours ahead\n")
        f.write(f"Lag: {LAG} steps, runs: {N_RUNS}, trials: {N_OPTUNA_TRIALS}\n")
        f.write(f"Persistence test NMSE: {persistence:.6f}\n\n")
        for m in MODELS:
            f.write(f"{m}:\n")
            f.write(f"  test NMSE mean +/- std: {run_means[m].mean():.6f} +/- {run_means[m].std():.6f}\n")
            f.write(f"  test NMSE min / max:    {run_means[m].min():.6f} / {run_means[m].max():.6f}\n\n")
        f.write(f"Welch t-test: t={t_stat:.3f}, p={p_value:.4f} "
                f"({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05)\n")
        f.write(f"Lower test NMSE: {better}\n\n")
        f.write("=== Per-run test NMSE ===\n")
        for r in run_results:
            f.write(f"  run {r['run_index']} (optuna_seed {r['optuna_seed']}): "
                    f"feedback {r['feedback']['test']:.6f}, customizable {r['customizable']['test']:.6f}\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
