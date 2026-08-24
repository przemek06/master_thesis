import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = os.path.dirname(__file__)
START = 0
WINDOW = 168

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
n_runs = len(run_means["customizable"])
test_target = arrays["test_target"]

seg = slice(START, START + WINDOW)
hours = np.arange(START, START + WINDOW)
for m in MODELS:
    pred = arrays[f"pred_{m}"]
    fig, ax = plt.subplots(figsize=WIDE)
    ax.plot(hours, test_target[seg], color=REFERENCE, linewidth=1)
    ax.plot(hours, pred[seg], color=COLORS[m], linestyle="--")
    ax.set_xlabel("hours")
    ax.set_ylabel("standardized temperature")
    save(fig, f"predictions_{m}.png")

idx = np.arange(n_runs)
fig, ax = plt.subplots(figsize=WIDE)
for m, offset in zip(MODELS, (-BAR_WIDTH / 2, BAR_WIDTH / 2)):
    ax.bar(idx + offset, run_means[m], BAR_WIDTH, color=COLORS[m])
ax.set_xticks(idx)
ax.set_xlabel("run")
ax.set_ylabel("test nmse")
save(fig, "test_nmse_bar.png")

lo = min(run_means[m].min() for m in MODELS) - 0.01
hi = max(run_means[m].max() for m in MODELS) + 0.01
grid = np.linspace(lo, hi, 300)
fig, ax = plt.subplots(figsize=WIDE)
for m in MODELS:
    kde = gaussian_kde(run_means[m], bw_method="scott")
    ax.plot(grid, kde(grid), color=COLORS[m])
    ax.fill_between(grid, kde(grid), color=COLORS[m], alpha=0.15)
ax.set_xlabel("test nmse")
ax.set_ylabel("density")
save(fig, "test_score_distribution.png")

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
