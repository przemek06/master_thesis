import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(__file__)
PLOT_DELAY = 5

DPI = 150
SQUARE = (6, 6)
WIDE = (7, 5)
PRIMARY = "darkorange"
SECONDARY = "steelblue"
REFERENCE = "black"
POINT_SIZE = 8
POINT_ALPHA = 0.6
BAR_WIDTH = 0.4

MODELS = ["customizable", "feedback"]
COLORS = {"customizable": PRIMARY, "feedback": SECONDARY}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, name), dpi=DPI)
    plt.close(fig)


arrays = np.load(os.path.join(HERE, "arrays.npz"))

run_means = {m: arrays[f"run_means_{m}"] for m in MODELS}
curves = {m: arrays[f"forgetting_curves_{m}"] for m in MODELS}
n_runs = len(run_means["customizable"])
k_max = curves["customizable"].shape[1]
k = np.arange(1, k_max + 1)

fig, ax = plt.subplots(figsize=WIDE)
for m in MODELS:
    mean_curve = curves[m].mean(axis=0)
    std_curve = curves[m].std(axis=0)
    ax.plot(k, mean_curve, color=COLORS[m])
    ax.fill_between(k, mean_curve - std_curve, mean_curve + std_curve, color=COLORS[m], alpha=0.2)
ax.set_xlabel("delay")
ax.set_ylabel("squared correlation")
save(fig, "forgetting_curve.png")

idx = np.arange(n_runs)
fig, ax = plt.subplots(figsize=WIDE)
for m, offset in zip(MODELS, (-BAR_WIDTH / 2, BAR_WIDTH / 2)):
    ax.bar(idx + offset, run_means[m], BAR_WIDTH, color=COLORS[m])
ax.set_xticks(idx)
ax.set_xlabel("run")
ax.set_ylabel("test memory capacity")
save(fig, "mc_bar.png")

lo = min(run_means[m].min() for m in MODELS) - 5
hi = max(run_means[m].max() for m in MODELS) + 5
grid = np.linspace(lo, hi, 300)
fig, ax = plt.subplots(figsize=WIDE)
for m in MODELS:
    kde = gaussian_kde(run_means[m], bw_method="scott")
    ax.plot(grid, kde(grid), color=COLORS[m])
    ax.fill_between(grid, kde(grid), color=COLORS[m], alpha=0.15)
ax.set_xlabel("test memory capacity")
ax.set_ylabel("density")
save(fig, "mc_distribution.png")

for m in MODELS:
    target = arrays[f"target_{m}"][:, PLOT_DELAY]
    pred = arrays[f"pred_{m}"][:, PLOT_DELAY]
    fig, ax = plt.subplots(figsize=SQUARE)
    lo = min(target.min(), pred.min())
    hi = max(target.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color=REFERENCE, linewidth=1)
    ax.scatter(target, pred, s=POINT_SIZE, alpha=POINT_ALPHA, color=COLORS[m])
    ax.set_aspect("equal")
    ax.set_xlabel("target")
    ax.set_ylabel("prediction")
    save(fig, f"target_vs_prediction_{m}.png")

theta = np.linspace(0, 2 * np.pi, 500)
for m in MODELS:
    W = arrays[f"W_{m}"]
    for r in range(n_runs):
        eigs = np.linalg.eigvals(W[r])
        fig, ax = plt.subplots(figsize=SQUARE)
        ax.plot(np.cos(theta), np.sin(theta), color=REFERENCE, linewidth=1)
        ax.scatter(eigs.real, eigs.imag, s=POINT_SIZE, alpha=POINT_ALPHA, color=COLORS[m])
        ax.set_aspect("equal")
        ax.set_xlabel("real")
        ax.set_ylabel("imaginary")
        save(fig, f"eigenvalues_{m}_run{r}.png")
