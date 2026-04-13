import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import csv
import numpy as np
import matplotlib.pyplot as plt
import optuna

from tasks.narma10 import load as load_narma10
from tasks.narma30 import load as load_narma30
from tasks.mackey_glass import load as load_mackey_glass
from tasks.lorenz import load as load_lorenz
from models.esn import ESN
from models.custom_esn_5 import CustomESN5
from generate import precompute_sparse_schur

HERE = os.path.dirname(__file__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

WARMUP = 100
N_TRIALS = 100
N_REFINE = 30
SIZES = [100, 200, 500, 1000]
CUSTOM_SPARSITY = {100: 0.95, 200: 0.95, 500: 0.98, 1000: 0.98}

TASKS = {
    "narma10":     (load_narma10,     False),
    "narma30":     (load_narma30,     False),
    "mackey_glass":(load_mackey_glass, True),
    "lorenz":      (load_lorenz,       True),
}


def nmse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2) / np.var(y_true)


def out_dir(*parts):
    path = os.path.join(HERE, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def save_trajectory(model, u_test, y_test, autonomous, path, title):
    if autonomous:
        y_pred = model.predict_autonomous(u_test[:WARMUP], len(u_test) - WARMUP)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        for i, ax in enumerate(axes):
            if i == 0:
                warmup_preds = model.predict(u_test[:WARMUP])
                full_pred = np.concatenate([warmup_preds, y_pred])
                ax.axvspan(0, WARMUP, alpha=0.15, color="gray", label="Warmup")
                ax.plot(np.concatenate([y_test[:WARMUP], y_test[WARMUP:WARMUP + 200]])[:WARMUP + 200],
                        label="Target", color="steelblue")
                ax.plot(full_pred[:WARMUP + 200], label="Prediction", color="tomato", linestyle="--")
                ax.set_title(f"Timesteps 0–{WARMUP + 200} (warmup shaded)")
            else:
                start = i * 400
                end = start + 200
                ax.plot(y_test[WARMUP + start:WARMUP + end], label="Target", color="steelblue")
                ax.plot(y_pred[start:end], label="Prediction", color="tomato", linestyle="--")
                ax.set_title(f"Prediction timesteps {start}–{end}")
            ax.legend(loc="upper right")
    else:
        y_pred = model.predict(u_test)
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        for i, ax in enumerate(axes):
            start = i * 100
            end = start + 200
            ax.plot(y_test[start:end], label="Target", color="steelblue")
            ax.plot(y_pred[start:end], label="Prediction", color="tomato", linestyle="--")
            ax.set_title(f"Timesteps {start}–{end}")
            ax.legend(loc="upper right")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(path, "trajectory.png"), dpi=150)
    plt.close(fig)


def save_eigenvalues(model, path, title):
    W_np = model.W.cpu().numpy()
    eigs = np.linalg.eigvals(W_np)
    theta = np.linspace(0, 2 * np.pi, 500)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
    ax.scatter(eigs.real, eigs.imag, s=10, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "eigenvalues.png"), dpi=150)
    plt.close(fig)


def load_result(path):
    params_file = os.path.join(path, "params.txt")
    if not os.path.exists(params_file):
        return None
    with open(params_file) as f:
        lines = f.readlines()
    val_nmse = float(lines[0].split(":")[1].strip())
    test_nmse = float(lines[1].split(":")[1].strip())
    return val_nmse, test_nmse


def make_esn_objective(u_train, y_train, u_val, y_val, size, autonomous):
    def objective(trial):
        params = {
            "spectral_radius": trial.suggest_float("spectral_radius", 0.1, 2.0),
            "sparsity":        trial.suggest_float("sparsity", 0.5, 0.99),
            "input_scaling":   trial.suggest_float("input_scaling", 0.1, 2.0),
            "leaky_rate":      trial.suggest_float("leaky_rate", 0.1, 1.0),
            "ridge":           trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
        }
        if autonomous:
            params["noise"] = trial.suggest_float("noise", 1e-5, 1e-1, log=True)
        model = ESN(n_inputs=1, n_reservoir=size, n_outputs=1, seed=0, **params)
        model.fit(u_train, y_train, warmup=WARMUP)
        if autonomous:
            y_pred = model.predict_autonomous(u_val[:WARMUP], len(u_val) - WARMUP)
            return nmse(y_val[WARMUP:], y_pred)
        else:
            return nmse(y_val, model.predict(u_val))
    return objective


def make_custom_objective(u_train, y_train, u_val, y_val, size, autonomous,
                          T_template, Z_template, sparsity_mask):
    def objective(trial):
        params = {
            "r_min":         trial.suggest_float("r_min", 0.0, 0.5),
            "r_max":         trial.suggest_float("r_max", 0.5, 1.5),
            "alpha":         trial.suggest_float("alpha", 0.1, 10.0),
            "input_scaling": trial.suggest_float("input_scaling", 0.1, 2.0),
            "leaky_rate":    trial.suggest_float("leaky_rate", 0.1, 1.0),
            "ridge":         trial.suggest_float("ridge", 1e-9, 1e-1, log=True),
        }
        if autonomous:
            params["noise"] = trial.suggest_float("noise", 1e-5, 1e-1, log=True)
        model = CustomESN5(n_inputs=1, n_reservoir=size, n_outputs=1,
                           T_template=T_template, Z_template=Z_template,
                           sparsity_mask=sparsity_mask, n_refine=N_REFINE, seed=0, **params)
        model.fit(u_train, y_train, warmup=WARMUP)
        if autonomous:
            y_pred = model.predict_autonomous(u_val[:WARMUP], len(u_val) - WARMUP)
            return nmse(y_val[WARMUP:], y_pred)
        else:
            return nmse(y_val, model.predict(u_val))
    return objective


def all_custom_cached(size):
    return all(
        os.path.exists(os.path.join(HERE, "custom_esn5", task_name, f"size_{size}", "params.txt"))
        for task_name in TASKS
    )

print("Precomputing Schur templates...")
schur_templates = {}
for size in SIZES:
    sparsity = CUSTOM_SPARSITY[size]
    rng = np.random.default_rng(42)
    sparsity_mask = rng.random((size, size)) > sparsity
    if all_custom_cached(size):
        print(f"  size={size} — all CustomESN5 results cached, skipping precompute")
        schur_templates[size] = (sparsity_mask, None, None)
    else:
        print(f"  size={size}, sparsity={sparsity}...")
        T_template, Z_template = precompute_sparse_schur(sparsity_mask, n_iter=100, seed=42)
        schur_templates[size] = (sparsity_mask, T_template, Z_template)

print("Loading tasks...")
task_data = {name: loader() for name, (loader, _) in TASKS.items()}

results = {}

for task_name, (_, autonomous) in TASKS.items():
    u_train, y_train, u_val, y_val, u_test, y_test = task_data[task_name]

    for size in SIZES:
        print(f"\n=== ESN | {task_name} | size={size} ===")
        path = out_dir("esn", task_name, f"size_{size}")
        cached = load_result(path)
        if cached:
            results[("esn", task_name, size)] = cached
            print(f"Skipped (cached)  Val NMSE: {cached[0]:.6f}  Test NMSE: {cached[1]:.6f}")
        else:
            study = optuna.create_study(direction="minimize")
            study.optimize(
                make_esn_objective(u_train, y_train, u_val, y_val, size, autonomous),
                n_trials=N_TRIALS, show_progress_bar=True,
            )
            best = study.best_params
            model = ESN(n_inputs=1, n_reservoir=size, n_outputs=1, seed=0, **best)
            model.fit(u_train, y_train, warmup=WARMUP)
            if autonomous:
                test_nmse = nmse(y_test[WARMUP:],
                                 model.predict_autonomous(u_test[:WARMUP], len(u_test) - WARMUP))
            else:
                test_nmse = nmse(y_test, model.predict(u_test))
            results[("esn", task_name, size)] = (study.best_value, test_nmse)
            with open(os.path.join(path, "params.txt"), "w") as f:
                f.write(f"Val NMSE:  {study.best_value:.6f}\n")
                f.write(f"Test NMSE: {test_nmse:.6f}\n")
                f.write(f"Best params: {best}\n")
            save_trajectory(model, u_test, y_test, autonomous, path,
                            f"ESN — {task_name} size={size}")
            save_eigenvalues(model, path, f"Eigenvalues — ESN {task_name} size={size}")
            print(f"Val NMSE: {study.best_value:.6f}  Test NMSE: {test_nmse:.6f}")

        print(f"\n=== CustomESN5 | {task_name} | size={size} ===")
        path = out_dir("custom_esn5", task_name, f"size_{size}")
        cached = load_result(path)
        if cached:
            results[("custom_esn5", task_name, size)] = cached
            print(f"Skipped (cached)  Val NMSE: {cached[0]:.6f}  Test NMSE: {cached[1]:.6f}")
        else:
            sparsity_mask, T_template, Z_template = schur_templates[size]
            study = optuna.create_study(direction="minimize")
            study.optimize(
                make_custom_objective(u_train, y_train, u_val, y_val, size, autonomous,
                                      T_template, Z_template, sparsity_mask),
                n_trials=N_TRIALS, show_progress_bar=True,
            )
            best = study.best_params
            model = CustomESN5(n_inputs=1, n_reservoir=size, n_outputs=1,
                               T_template=T_template, Z_template=Z_template,
                               sparsity_mask=sparsity_mask, n_refine=N_REFINE, seed=0, **best)
            model.fit(u_train, y_train, warmup=WARMUP)
            if autonomous:
                test_nmse = nmse(y_test[WARMUP:],
                                 model.predict_autonomous(u_test[:WARMUP], len(u_test) - WARMUP))
            else:
                test_nmse = nmse(y_test, model.predict(u_test))
            results[("custom_esn5", task_name, size)] = (study.best_value, test_nmse)
            with open(os.path.join(path, "params.txt"), "w") as f:
                f.write(f"Val NMSE:  {study.best_value:.6f}\n")
                f.write(f"Test NMSE: {test_nmse:.6f}\n")
                f.write(f"Best params: {best}\n")
            save_trajectory(model, u_test, y_test, autonomous, path,
                            f"CustomESN5 — {task_name} size={size}")
            save_eigenvalues(model, path, f"Eigenvalues — CustomESN5 {task_name} size={size}")
            print(f"Val NMSE: {study.best_value:.6f}  Test NMSE: {test_nmse:.6f}")

with open(os.path.join(HERE, "results.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "task", "size", "val_nmse", "test_nmse"])
    for (model_name, task_name, size), (val_nmse, test_nmse) in sorted(results.items()):
        writer.writerow([model_name, task_name, size, f"{val_nmse:.6f}", f"{test_nmse:.6f}"])

for task_name in TASKS:
    for split, idx, ylabel in [("val", 0, "Val NMSE"), ("test", 1, "Test NMSE")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, color, label in [("esn", "steelblue", "ESN"), ("custom_esn5", "tomato", "CustomESN5")]:
            nmse_vals = [results[(model_name, task_name, size)][idx] for size in SIZES]
            ax.plot(SIZES, nmse_vals, marker="o", color=color, label=label)
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels([str(s) for s in SIZES])
        ax.set_xlabel("Reservoir size")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs reservoir size — {task_name}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, f"nmse_{split}_{task_name}.png"), dpi=150)
        plt.close(fig)
