import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
SPARSITY = 0.95
ITERATIONS = 50
SEED = 0
SIZES = [100, 500, 1000, 2000]

CONFIGS = {
    "small_ring": {"r_min": 0.25, "r_max": 0.30, "alpha": 1.0},
    "medium_ring": {"r_min": 0.55, "r_max": 0.60, "alpha": 1.0},
    "wide_ring": {"r_min": 0.85, "r_max": 0.90, "alpha": 1.0},
    "wide_annulus": {"r_min": 0.30, "r_max": 0.90, "alpha": 1.0},
    "uniform_disk": {"r_min": 0.00, "r_max": 0.90, "alpha": 1.0},
}

DPI = 150
WIDE = (7, 5)
POINT_SIZE = 8
SERIES_COLORS = plt.cm.viridis(np.linspace(0.1, 0.85, len(CONFIGS)))


def label(name):
    return name.replace("_", " ")


def construction_time(cfg, size):
    def eigenvalue_fn(n):
        return sample_eigenvalues_ginibre(size=n, **cfg)

    t0 = time.time()
    generate_isospectral_sparse_matrix(
        eigenvalue_fn, size, SPARSITY, iterations=ITERATIONS, threshold=0.0, seed=SEED
    )
    return time.time() - t0


times = {}
for name, cfg in CONFIGS.items():
    times[name] = {}
    for size in SIZES:
        elapsed = construction_time(cfg, size)
        times[name][size] = elapsed
        print(f"{name} N={size}: {elapsed:.2f}s", flush=True)

config = {
    "method": "generate_isospectral_sparse_matrix + sample_eigenvalues_ginibre",
    "sparsity": SPARSITY,
    "iterations": ITERATIONS,
    "seed": SEED,
    "sizes": SIZES,
}

with open(os.path.join(HERE, "construction_times.json"), "w") as f:
    json.dump({"config": config, "times": times}, f, indent=2)

with open(os.path.join(HERE, "construction_times.txt"), "w") as f:
    f.write(f"sparsity={SPARSITY}, iterations={ITERATIONS}, seed={SEED}\n")
    f.write("\n=== Construction time in seconds ===\n\n")
    header = "config".ljust(14) + "".join(f"N={s}".rjust(12) for s in SIZES)
    f.write(header + "\n")
    for name in CONFIGS:
        row = label(name).ljust(14) + "".join(f"{times[name][s]:.2f}".rjust(12) for s in SIZES)
        f.write(row + "\n")

fig, ax = plt.subplots(figsize=WIDE)
for (name, color) in zip(CONFIGS, SERIES_COLORS):
    ax.plot(SIZES, [times[name][s] for s in SIZES], color=color, marker="o",
            markersize=POINT_SIZE / 2, label=label(name))
ax.set_xlabel("reservoir size")
ax.set_ylabel("construction time in seconds")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(HERE, "construction_times.png"), dpi=DPI)
plt.close(fig)

print("Done", flush=True)
