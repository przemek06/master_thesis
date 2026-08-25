import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import copy
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
WARMUP = 1000
DIVERGENCE_WINDOW = 20
DIVERGENCE_THRESHOLD = 1.0
N_RUNS = 20
N_VAL = 5
N_TEST = 20
N_OPTUNA_TRIALS = 75
N_RESERVOIR = 400
N_INPUTS = 1
N_OUTPUTS = 1
SEED = 0
N_WORKERS = 5
MODELS = ["feedback", "customizable"]


def fixed_weights():
    rng = np.random.default_rng(SEED)
    W_in = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
    W_fb = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))
    return W_in, W_fb


def denormalize(x):
    return np.arctanh(np.clip(x, -1 + 1e-7, 1 - 1e-7)) + 1


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def steps_until_divergence(pred, target):
    for i in range(DIVERGENCE_WINDOW, len(pred) + 1):
        if nmse(pred[i - DIVERGENCE_WINDOW:i], target[i - DIVERGENCE_WINDOW:i]) > DIVERGENCE_THRESHOLD:
            return i - DIVERGENCE_WINDOW
    return len(pred)


def eval_steps(model, series_list):
    scores = []
    for u, y in series_list:
        model.predict(u[:WARMUP])
        auto_pred = denormalize(model.predict_autonomous(len(u) - WARMUP))
        scores.append(steps_until_divergence(auto_pred, denormalize(y[WARMUP:])))
    return scores


def series_stats(scores):
    arr = np.array(scores)
    return {"scores": [int(s) for s in scores], "mean": float(arr.mean()), "std": float(arr.std())}


def all_predictions(model, series_list):
    targets, preds = [], []
    for u, y in series_list:
        warmup_pred = denormalize(np.asarray(model.predict(u[:WARMUP])).reshape(-1))
        auto_pred = denormalize(np.asarray(model.predict_autonomous(len(u) - WARMUP)).reshape(-1))
        preds.append(np.concatenate([warmup_pred, auto_pred]).astype(np.float32))
        targets.append(denormalize(np.asarray(y).reshape(-1)).astype(np.float32))
    return np.stack(targets), np.stack(preds)


def build_feedback(trial, W_in, W_fb):
    from models.esn_feedback import ESNFeedback

    return ESNFeedback(
        n_inputs=N_INPUTS,
        n_reservoir=N_RESERVOIR,
        n_outputs=N_OUTPUTS,
        spectral_radius=trial.suggest_float("spectral_radius", 0.5, 1.0),
        sparsity=trial.suggest_float("sparsity", 0.95, 0.99),
        leaky_rate=trial.suggest_float("leaky_rate", 0.5, 1.0),
        ridge=trial.suggest_float("ridge", 1e-6, 1e-1, log=True),
        noise=trial.suggest_float("noise", 1e-6, 1e-2, log=True),
        input_scaling=trial.suggest_float("input_scaling", 0.1, 5.0),
        feedback_scaling=trial.suggest_float("feedback_scaling", 0.0, 1.0),
        W_in=W_in,
        W_fb=W_fb,
        seed=SEED,
    )


def build_customizable(trial, W_in, W_fb):
    from models.esn_customizable import ESNCustomizable
    from generate import generate_isospectral_sparse_matrix
    from distribution import sample_eigenvalues_ginibre

    r_min = trial.suggest_float("r_min", 0.0, 0.95)
    r_max = trial.suggest_float("r_max", r_min, 1.0)
    alpha = trial.suggest_float("alpha", 0.1, 10.0)
    sparsity = trial.suggest_float("sparsity", 0.9, 0.99)
    np.random.seed(SEED)
    W = generate_isospectral_sparse_matrix(
        lambda size: sample_eigenvalues_ginibre(r_min=r_min, r_max=r_max, alpha=alpha, size=size),
        N_RESERVOIR,
        sparsity,
        iterations=50,
        seed=SEED,
    )
    return ESNCustomizable(
        n_inputs=N_INPUTS,
        n_reservoir=N_RESERVOIR,
        n_outputs=N_OUTPUTS,
        leaky_rate=trial.suggest_float("leaky_rate", 0.1, 1.0),
        ridge=trial.suggest_float("ridge", 1e-6, 1e-1, log=True),
        noise=trial.suggest_float("noise", 1e-6, 1e-2, log=True),
        input_scaling=trial.suggest_float("input_scaling", 0.1, 5.0),
        feedback_scaling=trial.suggest_float("feedback_scaling", 0.0, 1.0),
        W_in=W_in,
        W=W,
        W_fb=W_fb,
        bias=np.array([0.2]),
        seed=SEED,
    )


BUILDERS = {"feedback": build_feedback, "customizable": build_customizable}


def run_one(task):
    model_name, run_idx = task
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    import optuna
    from tasks.mackey_glass_multi import load

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    build = BUILDERS[model_name]
    W_in, W_fb = fixed_weights()
    u_train, y_train, val_list, test_list = load(n_val=N_VAL, n_test=N_TEST)
    optuna_seed = run_idx

    holder = {"model": None, "score": -np.inf}

    def objective(trial):
        model = build(trial, W_in, W_fb)
        model.fit(u_train, y_train, warmup=WARMUP)
        score = float(np.mean(eval_steps(model, val_list)))
        if score > holder["score"]:
            holder["score"] = score
            holder["model"] = copy.deepcopy(model)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=optuna_seed))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    model = holder["model"]
    val_scores = eval_steps(model, val_list)
    test_scores = eval_steps(model, test_list)
    test_targets, test_preds = all_predictions(model, test_list)
    median_idx = int(np.argsort(test_scores)[len(test_scores) // 2])

    print(f"  {model_name} run {run_idx}: val {np.mean(val_scores):.1f} test {np.mean(test_scores):.1f}", flush=True)
    return {
        "model": model_name,
        "run_index": run_idx,
        "optuna_seed": optuna_seed,
        "best_val_score_optuna": float(study.best_value),
        "val": series_stats(val_scores),
        "test": series_stats(test_scores),
        "best_params": {k: float(v) for k, v in study.best_params.items()},
        "W": model.W.detach().cpu().numpy(),
        "target": test_targets[median_idx],
        "pred": test_preds[median_idx],
        "steps": int(test_scores[median_idx]),
        "test_targets": test_targets,
        "test_preds": test_preds,
        "test_steps": np.array(test_scores, dtype=np.int32),
    }


def main():
    tasks = [(m, r) for m in MODELS for r in range(N_RUNS)]
    print(f"Running {len(tasks)} tasks ({N_RUNS} runs x {len(MODELS)} models) on {N_WORKERS} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        outputs = list(executor.map(run_one, tasks))

    by_model = {m: sorted([o for o in outputs if o["model"] == m], key=lambda o: o["run_index"]) for m in MODELS}
    run_means = {m: np.array([o["test"]["mean"] for o in by_model[m]]) for m in MODELS}
    median_run = {m: int(np.argsort(run_means[m])[N_RUNS // 2]) for m in MODELS}

    t_stat, p_value = stats.ttest_rel(run_means["feedback"], run_means["customizable"])
    better = "esn_customizable" if run_means["customizable"].mean() > run_means["feedback"].mean() else "esn_feedback"

    W_stacks = {m: np.stack([o["W"] for o in by_model[m]]) for m in MODELS}
    all_target = {m: np.stack([o["test_targets"] for o in by_model[m]]) for m in MODELS}
    all_pred = {m: np.stack([o["test_preds"] for o in by_model[m]]) for m in MODELS}
    all_steps = {m: np.stack([o["test_steps"] for o in by_model[m]]) for m in MODELS}
    np.savez_compressed(
        os.path.join(HERE, "arrays.npz"),
        W_feedback=W_stacks["feedback"],
        W_customizable=W_stacks["customizable"],
        run_means_feedback=run_means["feedback"],
        run_means_customizable=run_means["customizable"],
        median_run_feedback=median_run["feedback"],
        median_run_customizable=median_run["customizable"],
        target_feedback=by_model["feedback"][median_run["feedback"]]["target"],
        pred_feedback=by_model["feedback"][median_run["feedback"]]["pred"],
        steps_feedback=by_model["feedback"][median_run["feedback"]]["steps"],
        target_customizable=by_model["customizable"][median_run["customizable"]]["target"],
        pred_customizable=by_model["customizable"][median_run["customizable"]]["pred"],
        steps_customizable=by_model["customizable"][median_run["customizable"]]["steps"],
        all_target_feedback=all_target["feedback"],
        all_pred_feedback=all_pred["feedback"],
        all_steps_feedback=all_steps["feedback"],
        all_target_customizable=all_target["customizable"],
        all_pred_customizable=all_pred["customizable"],
        all_steps_customizable=all_steps["customizable"],
    )

    run_results = []
    for r in range(N_RUNS):
        fb = by_model["feedback"][r]
        cu = by_model["customizable"][r]
        run_results.append({
            "run_index": r,
            "optuna_seed": fb["optuna_seed"],
            "esn_feedback": {k: fb[k] for k in ("best_val_score_optuna", "val", "test", "best_params")},
            "esn_customizable": {k: cu[k] for k in ("best_val_score_optuna", "val", "test", "best_params")},
        })

    results = {
        "config": {
            "n_runs": N_RUNS,
            "n_val": N_VAL,
            "n_test": N_TEST,
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_reservoir": N_RESERVOIR,
            "warmup": WARMUP,
            "seed": SEED,
            "divergence_window": DIVERGENCE_WINDOW,
            "divergence_threshold": DIVERGENCE_THRESHOLD,
            "n_workers": N_WORKERS,
        },
        "runs": run_results,
        "summary": {
            "esn_feedback": {
                "run_means": run_means["feedback"].tolist(),
                "mean_of_means": float(run_means["feedback"].mean()),
                "std_of_means": float(run_means["feedback"].std()),
            },
            "esn_customizable": {
                "run_means": run_means["customizable"].tolist(),
                "mean_of_means": float(run_means["customizable"].mean()),
                "std_of_means": float(run_means["customizable"].std()),
            },
            "statistical_test": {
                "type": "paired_t_test",
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
        f.write("=== Summary ===\n")
        f.write(f"esn feedback     test mean of means: {run_means['feedback'].mean():.1f} +/- {run_means['feedback'].std():.1f}\n")
        f.write(f"esn customizable test mean of means: {run_means['customizable'].mean():.1f} +/- {run_means['customizable'].std():.1f}\n")
        f.write(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'} at alpha=0.05)\n")
        f.write(f"Better model: {better}\n")
        f.write("\n=== Per-run results ===\n")
        for r in run_results:
            f.write(f"\nRun {r['run_index']} (optuna_seed={r['optuna_seed']})\n")
            f.write(f"  esn feedback     val mean: {r['esn_feedback']['val']['mean']:.1f}, test mean: {r['esn_feedback']['test']['mean']:.1f}\n")
            f.write(f"  esn customizable val mean: {r['esn_customizable']['val']['mean']:.1f}, test mean: {r['esn_customizable']['test']['mean']:.1f}\n")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
