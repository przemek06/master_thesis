import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import matplotlib.pyplot as plt
from tasks.narma30_batch import load
from models.esn_feedback import ESNFeedback
from models.esn_customizable import ESNCustomizable
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
WARMUP = 200
N_RESERVOIR = 400
N_INPUTS = 1
N_OUTPUTS = 1
N_TRAIN = 10
N_VAL = 10
N_TEST = 10
SEED = 0
WINDOW = 250


def nmse(pred, target):
    return np.mean((pred - target) ** 2) / np.var(target)


def eval_series(model, u, y):
    model.reset_state()
    model.predict(u[:WARMUP])
    pred = model.predict(u[WARMUP:]).reshape(-1)
    return pred, y[WARMUP:].reshape(-1)


rng = np.random.default_rng(SEED)
W_in_fixed = rng.choice([0.0, 0.14, -0.14], size=(N_RESERVOIR, N_INPUTS + 1), p=[0.5, 0.25, 0.25])
W_fb_fixed = rng.uniform(-1.0, 1.0, (N_RESERVOIR, N_OUTPUTS))

u_train_fb, y_train_fb, _, test_fb = load(n_val=N_VAL, n_test=N_TEST, n_train=N_TRAIN, seed_offset=0)
model_fb = ESNFeedback(
    n_inputs=N_INPUTS,
    n_reservoir=N_RESERVOIR,
    n_outputs=N_OUTPUTS,
    spectral_radius=0.9830364566841865,
    sparsity=0.9542779923119186,
    leaky_rate=0.9941797777247978,
    ridge=2.8015844568485245e-05,
    noise=0.0003359272187610427,
    input_scaling=2.7144424490210373,
    feedback_scaling=0.09501033552657816,
    W_in=W_in_fixed,
    W_fb=W_fb_fixed,
    seed=SEED,
)
model_fb.fit_batch(u_train_fb, y_train_fb, warmup=WARMUP)

u_train_cu, y_train_cu, _, test_cu = load(n_val=N_VAL, n_test=N_TEST, n_train=N_TRAIN, seed_offset=120)
W_custom = generate_isospectral_sparse_matrix(
    lambda size: sample_eigenvalues_ginibre(r_min=0.8603358282469881, r_max=0.9606887219144012, alpha=2.0836770140323084, size=size),
    N_RESERVOIR,
    0.9273891575247095,
    iterations=50,
    seed=SEED,
)
model_cu = ESNCustomizable(
    n_inputs=N_INPUTS,
    n_reservoir=N_RESERVOIR,
    n_outputs=N_OUTPUTS,
    leaky_rate=0.9794537411280659,
    ridge=0.03595877825404108,
    noise=0.00015923437674019805,
    input_scaling=1.316705069056048,
    feedback_scaling=0.34162422818999405,
    W_in=W_in_fixed,
    W=W_custom,
    W_fb=W_fb_fixed,
    bias=np.array([0.2]),
    seed=SEED,
)
model_cu.fit_batch(u_train_cu, y_train_cu, warmup=WARMUP)

models = {
    "feedback": (model_fb, model_fb.W.cpu().numpy(), test_fb),
    "customizable": (model_cu, W_custom, test_cu),
}

scores = {}
for name, (model, W, test_list) in models.items():
    series_nmse = []
    preds = []
    for u, y in test_list:
        pred, target = eval_series(model, u, y)
        series_nmse.append(nmse(pred, target))
        preds.append((pred, target))
    scores[name] = float(np.mean(series_nmse))

    median_idx = int(np.argsort(series_nmse)[len(series_nmse) // 2])
    pred, target = preds[median_idx]
    error = np.abs(pred - target)
    step = np.arange(WARMUP, WARMUP + len(pred))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(step[:WINDOW], target[:WINDOW], color="darkorange", label="target")
    ax.plot(step[:WINDOW], pred[:WINDOW], color="steelblue", label="prediction")
    ax.set_xlabel("step")
    ax.set_ylabel("output")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"prediction_{name}.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    lo = min(target.min(), pred.min())
    hi = max(target.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.scatter(target, pred, s=8, alpha=0.6, color="darkorange")
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"scatter_{name}.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(step, error, color="darkorange", linewidth=1)
    ax.set_xlabel("step")
    ax.set_ylabel("absolute error")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"error_{name}.png"), dpi=150)
    plt.close(fig)

    eigs = np.linalg.eigvals(W)
    theta = np.linspace(0, 2 * np.pi, 500)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
    ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.6, color="darkorange")
    ax.set_aspect("equal")
    ax.set_xlabel("real")
    ax.set_ylabel("imaginary")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"eigenvalues_{name}.png"), dpi=150)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar([0], [scores["feedback"]], width=0.4, color="darkorange", label="feedback")
ax.bar([1], [scores["customizable"]], width=0.4, color="steelblue", label="customizable")
ax.set_xticks([0, 1])
ax.set_xticklabels(["feedback", "customizable"])
ax.set_ylabel("nmse")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(HERE, "nmse_bar.png"), dpi=150)
plt.close(fig)

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write("single instance NARMA-30 test NMSE (mean over 10 test series)\n")
    f.write(f"feedback: {scores['feedback']:.4f}\n")
    f.write(f"customizable: {scores['customizable']:.4f}\n")
