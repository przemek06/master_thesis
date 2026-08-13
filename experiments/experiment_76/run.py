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
SUBSAMPLE = 6
HORIZON = 24
LAG = 12
N_INPUTS = 13 + LAG
N_OUTPUTS = 1
READOUT_INPUTS = True
RESERVOIR_SEED = 0
OPTUNA_SEED = 0
N_WORKERS = 2
MODELS = ["ginibre", "feedback"]

DPI = 150
WIDE = (7, 5)
SQUARE = (6, 6)
TIMESERIES = (14, 4)
PRIMARY = "darkorange"
SECONDARY = "steelblue"
REFERENCE = "black"
COLORS = {"ginibre": PRIMARY, "feedback": SECONDARY}
POINT_SIZE = 8
POINT_ALPHA = 0.6


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


def run_one(model_name):
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
        return evaluate(model, val_list)[0]

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    best = study.best_params
    model = build(best)
    model.fit(u_train, y_train, warmup=WARMUP)
    model.noise = 0.0
    val, val_preds = evaluate(model, val_list)
    test, test_preds = evaluate(model, test_list)

    W = model.W.numpy()
    print(f"  {model_name}: val {val:.6f} test {test:.6f}", flush=True)
    return {
        "model": model_name,
        "val": val,
        "test": test,
        "best_params": best,
        "trials": [{"number": t.number, "value": t.value, "params": t.params} for t in study.trials],
        "val_pred": val_preds[0],
        "test_pred": test_preds[0],
        "W": W,
        "W_out": model.W_out.numpy(),
        "eigenvalues": np.linalg.eigvals(W),
    }


def main():
    print(f"Running {len(MODELS)} models on {N_WORKERS} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        outputs = {r["model"]: r for r in executor.map(run_one, MODELS)}

    from tasks.jena import load
    _, _, val_list, test_list = load(horizon=HORIZON, lag=LAG, subsample=SUBSAMPLE)
    persistence = persistence_nmse(test_list)

    _, y_val = val_list[0]
    _, y_test = test_list[0]
    val_target = y_val[WARMUP:, 0]
    test_target = y_test[WARMUP:, 0]
    test_persistence = y_test[WARMUP - HORIZON:-HORIZON, 0]

    arrays = {
        "val_target": val_target,
        "test_target": test_target,
        "test_persistence": test_persistence,
    }
    for m in MODELS:
        arrays[f"val_pred_{m}"] = outputs[m]["val_pred"][:, 0]
        arrays[f"test_pred_{m}"] = outputs[m]["test_pred"][:, 0]
        arrays[f"W_{m}"] = outputs[m]["W"]
        arrays[f"W_out_{m}"] = outputs[m]["W_out"]
        arrays[f"eigenvalues_{m}"] = outputs[m]["eigenvalues"]
    np.savez_compressed(os.path.join(HERE, "arrays.npz"), **arrays)

    results = {
        "config": {
            "task": "jena",
            "subsample": SUBSAMPLE,
            "sampling_interval_minutes": 10 * SUBSAMPLE,
            "horizon": HORIZON,
            "horizon_hours": 10 * SUBSAMPLE * HORIZON / 60,
            "lag": LAG,
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_reservoir": N_RESERVOIR,
            "iso_iterations": ISO_ITERATIONS,
            "readout_inputs": READOUT_INPUTS,
            "warmup": WARMUP,
            "reservoir_seed": RESERVOIR_SEED,
            "optuna_seed": OPTUNA_SEED,
            "note": "hourly sampling, single run per model; feedback disabled for both, noise disabled at evaluation",
        },
        "persistence_test_nmse": persistence,
        "models": {
            m: {
                "val_nmse": outputs[m]["val"],
                "test_nmse": outputs[m]["test"],
                "best_params": outputs[m]["best_params"],
                "trials": outputs[m]["trials"],
            }
            for m in MODELS
        },
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("=== Single run per model | jena hourly ===\n")
        f.write(f"Sampling: every {SUBSAMPLE} rows = {10 * SUBSAMPLE} minutes\n")
        f.write(f"Horizon: {HORIZON} steps = {10 * SUBSAMPLE * HORIZON / 60:.0f} hours ahead\n")
        f.write(f"Lag: {LAG} steps\n")
        f.write(f"Optuna seed: {OPTUNA_SEED}, trials: {N_OPTUNA_TRIALS}\n")
        f.write(f"Persistence test NMSE: {persistence:.6f}\n\n")
        for m in MODELS:
            f.write(f"{m}:\n")
            f.write(f"  val NMSE:  {outputs[m]['val']:.6f}\n")
            f.write(f"  test NMSE: {outputs[m]['test']:.6f}\n")
            f.write(f"  best params: {outputs[m]['best_params']}\n\n")

    for m in MODELS:
        print(f"{m}: test NMSE {outputs[m]['test']:.6f}")
    print(f"persistence: {persistence:.6f}")

    print("Plotting...", flush=True)

    for span, name in [(24, "prediction_24h.png"), (168, "prediction_week.png")]:
        fig, ax = plt.subplots(figsize=TIMESERIES)
        hours = np.arange(span)
        ax.plot(hours, test_target[:span], color=REFERENCE, linewidth=1, label="target")
        for m in MODELS:
            ax.plot(hours, arrays[f"test_pred_{m}"][:span], color=COLORS[m], linestyle="--", label=m)
        ax.set_xlabel("hours")
        ax.set_ylabel("standardized temperature")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, name), dpi=DPI)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=WIDE)
    for m in MODELS:
        ax.hist(np.abs(arrays[f"test_pred_{m}"] - test_target), bins=100, density=True,
                color=COLORS[m], alpha=0.6, label=m)
    ax.set_yscale("log")
    ax.set_xlabel("absolute error")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "error_distribution.png"), dpi=DPI)
    plt.close(fig)

    for m in MODELS:
        fig, ax = plt.subplots(figsize=SQUARE)
        ax.scatter(test_target, arrays[f"test_pred_{m}"], s=POINT_SIZE, alpha=POINT_ALPHA, color=COLORS[m])
        lo, hi = test_target.min(), test_target.max()
        ax.plot([lo, hi], [lo, hi], color=REFERENCE, linewidth=1)
        ax.set_aspect("equal")
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"predicted_vs_target_{m}.png"), dpi=DPI)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=WIDE)
    for m in MODELS:
        values = np.array([t["value"] for t in outputs[m]["trials"] if t["value"] is not None])
        ax.plot(np.minimum.accumulate(values), color=COLORS[m], label=m)
    ax.set_yscale("log")
    ax.set_xlabel("trial")
    ax.set_ylabel("best validation NMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "optimization_history.png"), dpi=DPI)
    plt.close(fig)

    theta = np.linspace(0, 2 * np.pi, 500)
    for m in MODELS:
        eigs = arrays[f"eigenvalues_{m}"]
        fig, ax = plt.subplots(figsize=SQUARE)
        ax.plot(np.cos(theta), np.sin(theta), color=REFERENCE, linewidth=1)
        ax.scatter(eigs.real, eigs.imag, s=POINT_SIZE, alpha=POINT_ALPHA, color=COLORS[m])
        ax.set_aspect("equal")
        ax.set_xlabel("real")
        ax.set_ylabel("imaginary")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"eigenvalues_{m}.png"), dpi=DPI)
        plt.close(fig)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
