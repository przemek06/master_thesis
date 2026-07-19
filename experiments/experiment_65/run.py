import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json

import numpy as np
import matplotlib.pyplot as plt
import optuna

from tasks.wth import load
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

WARMUP = 100
N_OPTUNA_TRIALS = 100
N_RESERVOIR = 400
ISO_ITERATIONS = 50
HORIZON = 24
LAG = 24
N_INPUTS = 10 + LAG
N_OUTPUTS = 1
READOUT_INPUTS = True
SEED = 0
PLOT_SHOW = 400


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def eval_nmse(model, series_list):
    scores = []
    preds = []
    for u, y in series_list:
        model.predict(u[:WARMUP])
        pred = model.predict(u[WARMUP:], initial_state=model.last_state)
        scores.append(nmse(pred, y[WARMUP:]))
        preds.append(pred)
    return float(np.mean(scores)), preds


def persistence_nmse(series_list):
    scores = [nmse(y[WARMUP - HORIZON:-HORIZON], y[WARMUP:]) for _, y in series_list]
    return float(np.mean(scores))


def make_W(params):
    return generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(
            r_min=params["r_min"], r_max=params["r_max"], alpha=params["alpha"], size=size
        ),
        N_RESERVOIR,
        params["sparsity"],
        iterations=ISO_ITERATIONS,
        seed=SEED,
    )


def build(params, W):
    return ESNCustomizable(
        n_inputs=N_INPUTS,
        n_reservoir=N_RESERVOIR,
        n_outputs=N_OUTPUTS,
        input_scaling=params["input_scaling"],
        leaky_rate=params["leaky_rate"],
        ridge=params["ridge"],
        noise=params["noise"],
        feedback_scaling=0.0,
        W=W,
        bias=np.array([0.2]),
        readout_inputs=READOUT_INPUTS,
        seed=SEED,
        device="cpu",
    )


def main():
    u_train, y_train, val_list, test_list = load(horizon=HORIZON, lag=LAG)

    def objective(trial):
        r_min = trial.suggest_float("r_min", 0.0, 0.95)
        params = {
            "r_min": r_min,
            "r_max": trial.suggest_float("r_max", r_min, 1.0),
            "alpha": trial.suggest_float("alpha", 0.1, 10.0),
            "sparsity": trial.suggest_float("sparsity", 0.9, 0.99),
            "input_scaling": trial.suggest_float("input_scaling", 0.1, 5.0),
            "leaky_rate": trial.suggest_float("leaky_rate", 0.01, 1.0, log=True),
            "ridge": trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
            "noise": trial.suggest_float("noise", 1e-6, 1e-2, log=True),
        }
        model = build(params, make_W(params))
        model.fit(u_train, y_train, warmup=WARMUP)
        score, _ = eval_nmse(model, val_list)
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_params
    W = make_W(best)
    model = build(best, W)
    model.fit(u_train, y_train, warmup=WARMUP)
    val_score, _ = eval_nmse(model, val_list)
    test_score, test_preds = eval_nmse(model, test_list)
    persistence = persistence_nmse(test_list)

    results = {
        "config": {
            "model": "ESNCustomizable (Ginibre isospectral W, feedback_scaling=0)",
            "task": "wth",
            "horizon": HORIZON,
            "lag": LAG,
            "readout_inputs": READOUT_INPUTS,
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_reservoir": N_RESERVOIR,
            "iso_iterations": ISO_ITERATIONS,
            "warmup": WARMUP,
            "seed": SEED,
        },
        "val_nmse": val_score,
        "test_nmse": test_score,
        "persistence_test_nmse": persistence,
        "best_params": best,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write(f"=== ESNCustomizable (Ginibre) | wth horizon={HORIZON} ===\n")
        f.write(f"Val NMSE:         {val_score:.6f}\n")
        f.write(f"Test NMSE:        {test_score:.6f}\n")
        f.write(f"Persistence NMSE: {persistence:.6f}\n")
        f.write(f"Best params: {best}\n")
    print(f"Val NMSE: {val_score:.6f}  Test NMSE: {test_score:.6f}  Persistence: {persistence:.6f}", flush=True)

    _, y_test = test_list[0]
    target = y_test[WARMUP:, 0]
    pred = test_preds[0][:, 0]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(target[:PLOT_SHOW], label="Target", color="steelblue")
    ax.plot(pred[:PLOT_SHOW], label="Prediction", color="tomato", linestyle="--", alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Standardized WetBulbCelsius")
    ax.set_title(f"ESNCustomizable (Ginibre) — WTH {HORIZON}-step forecast (test NMSE={test_score:.4f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "predictions.png"), dpi=150)
    plt.close()

    eigs = np.linalg.eigvals(W)
    theta = np.linspace(0, 2 * np.pi, 500)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
    ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title("ESNCustomizable (Ginibre) — Eigenvalues of W")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "eigenvalues.png"), dpi=150)
    plt.close()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
