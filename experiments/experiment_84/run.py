import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import copy
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats as sps

from tasks.memory_capacity import K_MAX

HERE = os.path.dirname(__file__)
WARMUP = 200
N_RUNS = 10
N_TRAIN = 10
N_VAL = 10
N_TEST = 10
N_OPTUNA_TRIALS = 100
N_RESERVOIR = 100
ISO_ITERATIONS = 50
N_INPUTS = 1
N_OUTPUTS = K_MAX
SEED = 0
OPTUNA_SEED = 0
N_WORKERS = 5
MODELS = ["feedback", "customizable"]


def fixed_weights():
    rng = np.random.default_rng(SEED)
    W_in = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
    W_fb = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))
    return W_in, W_fb


def eval_mc(model, series_list):
    totals, curves = [], []
    for u, y in series_list:
        pred = model.predict(u)[WARMUP:]
        target = y[WARMUP:]
        mc_k = np.array([np.corrcoef(pred[:, k], target[:, k])[0, 1] ** 2 for k in range(K_MAX)])
        curves.append(mc_k)
        totals.append(float(mc_k.sum()))
    return totals, curves


def shared_params(trial):
    return {
        "sparsity": trial.suggest_float("sparsity", 0.9, 0.99),
        "leaky_rate": trial.suggest_float("leaky_rate", 0.1, 1.0),
        "ridge": trial.suggest_float("ridge", 1e-6, 1e-1, log=True),
        "noise": trial.suggest_float("noise", 1e-6, 1e-2, log=True),
        "input_scaling": trial.suggest_float("input_scaling", 0.1, 5.0),
    }


def build_customizable(trial, W_in, W_fb):
    from models.esn_customizable import ESNCustomizable
    from generate import generate_isospectral_sparse_matrix
    from distribution import sample_eigenvalues_ginibre

    p = shared_params(trial)
    r_min = trial.suggest_float("r_min", 0.0, 0.95)
    r_max = trial.suggest_float("r_max", r_min, 1.0)
    alpha = trial.suggest_float("alpha", 0.1, 10.0)
    np.random.seed(SEED)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=r_min, r_max=r_max, alpha=alpha, size=size),
        N_RESERVOIR, p["sparsity"], iterations=ISO_ITERATIONS, seed=SEED,
    )
    model = ESNCustomizable(
        n_inputs=N_INPUTS, n_reservoir=N_RESERVOIR, n_outputs=N_OUTPUTS,
        leaky_rate=p["leaky_rate"], ridge=p["ridge"], noise=p["noise"],
        input_scaling=p["input_scaling"], feedback_scaling=0.0,
        W_in=W_in, W=W, W_fb=W_fb, bias=np.array([0.2]), seed=SEED, device="cpu",
    )
    return model, W


def build_feedback(trial, W_in, W_fb):
    from models.esn_feedback import ESNFeedback

    p = shared_params(trial)
    model = ESNFeedback(
        n_inputs=N_INPUTS, n_reservoir=N_RESERVOIR, n_outputs=N_OUTPUTS,
        spectral_radius=trial.suggest_float("spectral_radius", 0.5, 1.0),
        sparsity=p["sparsity"], leaky_rate=p["leaky_rate"], ridge=p["ridge"],
        noise=p["noise"], input_scaling=p["input_scaling"], feedback_scaling=0.0,
        W_in=W_in, W_fb=W_fb, seed=SEED, device="cpu",
    )
    return model, model.W.detach().cpu().numpy()


BUILDERS = {"customizable": build_customizable, "feedback": build_feedback}


def run_one(task):
    model_name, run_idx = task
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    import optuna
    from tasks.memory_capacity import load

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    build = BUILDERS[model_name]
    W_in, W_fb = fixed_weights()
    data_offset = run_idx * (N_TRAIN + N_VAL + N_TEST)
    u_train, y_train, val_list, test_list = load(
        n_val=N_VAL, n_test=N_TEST, n_train=N_TRAIN, seed_offset=data_offset
    )

    holder = {"model": None, "score": -np.inf, "W": None}

    def objective(trial):
        model, W = build(trial, W_in, W_fb)
        model.fit_batch(u_train, y_train, warmup=WARMUP)
        score = float(np.mean(eval_mc(model, val_list)[0]))
        if score > holder["score"]:
            holder["score"] = score
            holder["model"] = copy.deepcopy(model)
            holder["W"] = W
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=OPTUNA_SEED))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    model = holder["model"]
    val_totals, _ = eval_mc(model, val_list)
    test_totals, test_curves = eval_mc(model, test_list)
    curve = np.mean(test_curves, axis=0)

    u0, y0 = test_list[0]
    full_pred = model.predict(u0)[WARMUP:].astype(np.float32)
    full_target = y0[WARMUP:].astype(np.float32)

    print(f"  {model_name} run {run_idx}: val MC {np.mean(val_totals):.2f} test MC {np.mean(test_totals):.2f}", flush=True)
    return {
        "model": model_name,
        "run_index": run_idx,
        "data_offset": data_offset,
        "val_mc": float(np.mean(val_totals)),
        "test_mc": float(np.mean(test_totals)),
        "test_mc_std": float(np.std(test_totals)),
        "readout_width": int(model._state_size if hasattr(model, "_state_size") else N_RESERVOIR),
        "best_params": {k: float(v) for k, v in study.best_params.items()},
        "curve": curve,
        "W": holder["W"],
        "full_pred": full_pred,
        "full_target": full_target,
    }


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
    tasks = [(m, r) for m in MODELS for r in range(N_RUNS)]
    print(f"Running {len(tasks)} tasks ({N_RUNS} runs x {len(MODELS)} models) on {N_WORKERS} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        outputs = list(executor.map(run_one, tasks))

    by_model = {m: sorted([o for o in outputs if o["model"] == m], key=lambda o: o["run_index"]) for m in MODELS}
    test_mc = {m: np.array([o["test_mc"] for o in by_model[m]]) for m in MODELS}
    val_mc = {m: np.array([o["val_mc"] for o in by_model[m]]) for m in MODELS}
    curves = {m: np.stack([o["curve"] for o in by_model[m]]) for m in MODELS}
    widths = {m: by_model[m][0]["readout_width"] for m in MODELS}
    median_run = {m: int(np.argsort(test_mc[m])[N_RUNS // 2]) for m in MODELS}

    t_stat, p_value = sps.ttest_rel(test_mc["customizable"], test_mc["feedback"])
    better = "customizable" if test_mc["customizable"].mean() > test_mc["feedback"].mean() else "feedback"

    W_stacks = {m: np.stack([o["W"] for o in by_model[m]]) for m in MODELS}

    arrays = {}
    for m in MODELS:
        arrays[f"run_means_{m}"] = test_mc[m]
        arrays[f"val_means_{m}"] = val_mc[m]
        arrays[f"forgetting_curves_{m}"] = curves[m]
        arrays[f"W_{m}"] = W_stacks[m]
        arrays[f"median_run_{m}"] = median_run[m]
        arrays[f"pred_{m}"] = by_model[m][median_run[m]]["full_pred"]
        arrays[f"target_{m}"] = by_model[m][median_run[m]]["full_target"]
    np.savez_compressed(os.path.join(HERE, "arrays.npz"), **arrays)

    results = {
        "config": {
            "task": "memory_capacity",
            "k_max": K_MAX,
            "n_reservoir": N_RESERVOIR,
            "readout_width": widths,
            "n_runs": N_RUNS,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "iso_iterations": ISO_ITERATIONS,
            "warmup": WARMUP,
            "seed": SEED,
            "optuna_seed": OPTUNA_SEED,
            "n_workers": N_WORKERS,
            "note": "both models use 100 reservoir units, identical W_in and data offsets, and identical "
                    "ranges for every shared hyperparameter. MC is bounded by the readout width, which is "
                    "2x larger for the complex customizable state",
        },
        "customizable": stats(test_mc["customizable"]),
        "feedback": stats(test_mc["feedback"]),
        "statistical_test": {
            "type": "paired_t_test",
            "note": "paired: run i uses the same data offset for both models",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "better_model": better,
        },
        "runs": [
            {
                "run_index": r,
                "data_offset": by_model["feedback"][r]["data_offset"],
                "feedback": {k: by_model["feedback"][r][k] for k in ("val_mc", "test_mc", "test_mc_std", "best_params")},
                "customizable": {k: by_model["customizable"][r][k] for k in ("val_mc", "test_mc", "test_mc_std", "best_params")},
            }
            for r in range(N_RUNS)
        ],
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(HERE, "results.txt"), "w") as f:
        f.write("=== Memory capacity, matched reservoir size (higher is better) ===\n")
        f.write(f"Reservoir units: {N_RESERVOIR} for both models\n")
        for m in MODELS:
            f.write(f"Readout width ({m}): {widths[m]} -> MC bound {widths[m]}\n")
        f.write(f"K_MAX: {K_MAX}\n\n")
        for m in MODELS:
            s = stats(test_mc[m])
            f.write(f"{m}:\n")
            f.write(f"  test MC mean +/- std: {s['mean']:.2f} +/- {s['std']:.2f}\n")
            f.write(f"  test MC median:       {s['median']:.2f}\n")
            f.write(f"  test MC min / max:    {s['min']:.2f} / {s['max']:.2f}\n")
            f.write(f"  fraction of bound:    {s['mean'] / widths[m]:.3f}\n\n")
        f.write(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f} "
                f"({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05)\n")
        f.write(f"Higher test MC: {better}\n\n")
        f.write("=== Per-run test MC ===\n")
        for r in range(N_RUNS):
            f.write(f"  run {r} (offset {by_model['feedback'][r]['data_offset']}): "
                    f"customizable {test_mc['customizable'][r]:.2f}, feedback {test_mc['feedback'][r]:.2f}\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
