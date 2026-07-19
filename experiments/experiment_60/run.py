import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json

import numpy as np
import matplotlib.pyplot as plt
import optuna

from tasks.santafe_laser import load
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

WARMUP = 100
N_OPTUNA_TRIALS = 100
N_RESERVOIR = 400
ISO_ITERATIONS = 50
READOUT_INPUTS = True
SEED = 0
PLOT_START = 0
PLOT_SHOW = 400


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def eval_nmse(model, u, y):
    model.predict(u[:WARMUP])
    pred = model.predict(u[WARMUP:], initial_state=model.last_state)
    return nmse(pred, y[WARMUP:]), pred


def make_W(params):
    r_min, r_max, alpha, sparsity = params["r_min"], params["r_max"], params["alpha"], params["sparsity"]
    return generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=r_min, r_max=r_max, alpha=alpha, size=size),
        N_RESERVOIR,
        sparsity,
        iterations=ISO_ITERATIONS,
        seed=SEED,
    )


def build(params, W):
    return ESNCustomizable(
        n_inputs=1,
        n_reservoir=N_RESERVOIR,
        n_outputs=1,
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
    u_train, y_train, u_val, y_val, u_test, y_test = load()

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
        score, _ = eval_nmse(model, u_val, y_val)
        return score

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_params
    W = make_W(best)
    model = build(best, W)
    model.fit(u_train, y_train, warmup=WARMUP)
    val_nmse, _ = eval_nmse(model, u_val, y_val)
    test_nmse, test_pred = eval_nmse(model, u_test, y_test)
    persistence_nmse = nmse(u_test[WARMUP:], y_test[WARMUP:])

    results = {
        "config": {
            "model": "ESNCustomizable (Ginibre isospectral W, feedback_scaling=0)",
            "task": "santafe_laser",
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_reservoir": N_RESERVOIR,
            "iso_iterations": ISO_ITERATIONS,
            "readout_inputs": READOUT_INPUTS,
            "warmup": WARMUP,
            "seed": SEED,
        },
        "val_nmse": float(val_nmse),
        "test_nmse": float(test_nmse),
        "persistence_test_nmse": float(persistence_nmse),
        "best_params": best,
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("=== ESNCustomizable (Ginibre) | santafe_laser ===\n")
        f.write(f"Val NMSE:         {val_nmse:.6f}\n")
        f.write(f"Test NMSE:        {test_nmse:.6f}\n")
        f.write(f"Persistence NMSE: {persistence_nmse:.6f}\n")
        f.write(f"Best params: {best}\n")
    print(f"Val NMSE: {val_nmse:.6f}  Test NMSE: {test_nmse:.6f}  Persistence: {persistence_nmse:.6f}", flush=True)

    target = y_test[WARMUP:, 0]
    pred = test_pred[:, 0]
    sl = slice(PLOT_START, PLOT_START + PLOT_SHOW)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(target[sl], label="Target", color="steelblue")
    ax.plot(pred[sl], label="Prediction", color="tomato", linestyle="--", alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Standardized intensity")
    ax.set_title(f"ESNCustomizable (Ginibre) — Santa Fe laser one-step forecast (test NMSE={test_nmse:.4f})")
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
