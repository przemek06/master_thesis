import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json

import numpy as np
import matplotlib.pyplot as plt

from tasks.santafe_laser import load
from models.esn import ESN
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

WARMUP = 100
N_RESERVOIR = 400
ISO_ITERATIONS = 50
READOUT_INPUTS = True
SEEDS = list(range(10))
PLOT_SHOW = 400


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def eval_nmse(model, u, y):
    model.predict(u[:WARMUP])
    pred = model.predict(u[WARMUP:], initial_state=model.last_state)
    return nmse(pred, y[WARMUP:]), pred


def best_params(exp):
    with open(os.path.join(ROOT, exp, "results.json")) as f:
        return json.load(f)["best_params"]


def build_esn(p, seed):
    return ESN(
        n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1,
        spectral_radius=p["spectral_radius"], sparsity=p["sparsity"],
        input_scaling=p["input_scaling"], leaky_rate=p["leaky_rate"],
        ridge=p["ridge"], noise=p["noise"], readout_inputs=READOUT_INPUTS, seed=seed, device="cpu",
    )


def build_ginibre(p, seed):
    np.random.seed(seed)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=p["r_min"], r_max=p["r_max"], alpha=p["alpha"], size=size),
        N_RESERVOIR, p["sparsity"], iterations=ISO_ITERATIONS, seed=seed,
    )
    model = ESNCustomizable(
        n_inputs=1, n_reservoir=N_RESERVOIR, n_outputs=1,
        input_scaling=p["input_scaling"], leaky_rate=p["leaky_rate"],
        ridge=p["ridge"], noise=p["noise"], feedback_scaling=0.0,
        W=W, bias=np.array([0.2]), readout_inputs=READOUT_INPUTS, seed=seed, device="cpu",
    )
    return model


def run_model(name, builder, params, u_train, y_train, u_test, y_test):
    scores = []
    for seed in SEEDS:
        model = builder(params, seed)
        model.fit(u_train, y_train, warmup=WARMUP)
        score, _ = eval_nmse(model, u_test, y_test)
        scores.append(float(score))
        print(f"  {name} seed={seed}: test NMSE {score:.6f}", flush=True)
    return np.array(scores)


def stats(scores):
    return {
        "scores": scores.tolist(),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "median": float(np.median(scores)),
    }


def main():
    u_train, y_train, u_val, y_val, u_test, y_test = load()
    persistence = float(nmse(u_test[WARMUP:], y_test[WARMUP:]))

    print("ESN...", flush=True)
    esn_scores = run_model("ESN", build_esn, best_params("experiment_59"),
                           u_train, y_train, u_test, y_test)
    print("ESNCustomizable (Ginibre)...", flush=True)
    gin_scores = run_model("Ginibre", build_ginibre, best_params("experiment_60"),
                           u_train, y_train, u_test, y_test)

    results = {
        "config": {
            "task": "santafe_laser",
            "n_reservoir": N_RESERVOIR,
            "iso_iterations": ISO_ITERATIONS,
            "readout_inputs": READOUT_INPUTS,
            "warmup": WARMUP,
            "seeds": SEEDS,
            "note": "hyperparameters fixed from experiment_59/60, only reservoir seed varies",
        },
        "persistence_test_nmse": persistence,
        "esn": stats(esn_scores),
        "ginibre": stats(gin_scores),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("=== Reservoir-seed variance | santafe_laser ===\n")
        f.write(f"Seeds: {SEEDS}\n")
        f.write(f"Persistence test NMSE: {persistence:.6f}\n\n")
        for name, s in [("ESN", esn_scores), ("Ginibre", gin_scores)]:
            f.write(f"{name}:\n")
            f.write(f"  mean +/- std: {s.mean():.6f} +/- {s.std():.6f}\n")
            f.write(f"  median:       {np.median(s):.6f}\n")
            f.write(f"  min / max:    {s.min():.6f} / {s.max():.6f}\n\n")

    print(f"\nESN     test NMSE: {esn_scores.mean():.6f} +/- {esn_scores.std():.6f}")
    print(f"Ginibre test NMSE: {gin_scores.mean():.6f} +/- {gin_scores.std():.6f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [esn_scores, gin_scores]
    labels = ["ESN", "Ginibre"]
    colors = ["steelblue", "darkorange"]
    bp = ax.boxplot(data, labels=labels, showmeans=True, widths=0.5)
    for i, (scores, color) in enumerate(zip(data, colors)):
        x = np.random.default_rng(0).normal(i + 1, 0.05, len(scores))
        ax.scatter(x, scores, color=color, alpha=0.7, zorder=3, s=25)
    ax.set_ylabel("Test NMSE")
    ax.set_title(f"Test NMSE across {len(SEEDS)} reservoir seeds (fixed split & hyperparameters)")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "seed_variance.png"), dpi=150)
    plt.close()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
