import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
N_RESERVOIR = 400
SPARSITY = 0.95
ITERATIONS = 50
THRESHOLD = 0.01
SEED = 0

CONFIGS = {
    "small_ring": {"r_min": 0.25, "r_max": 0.30, "alpha": 1.0},
    "medium_ring": {"r_min": 0.55, "r_max": 0.60, "alpha": 1.0},
    "wide_ring": {"r_min": 0.85, "r_max": 0.90, "alpha": 1.0},
    "wide_annulus": {"r_min": 0.30, "r_max": 0.90, "alpha": 1.0},
    "uniform_disk": {"r_min": 0.00, "r_max": 0.90, "alpha": 1.0},
}

CANDIDATES = {"normal": stats.norm, "cauchy": stats.cauchy}

DPI = 150
SQUARE = (6, 6)
WIDE = (7, 5)
CMAP = "viridis"
PRIMARY = "darkorange"
SECONDARY = "steelblue"
REFERENCE = "black"
POINT_SIZE = 8
POINT_ALPHA = 0.6
BAR_WIDTH = 0.4
VMAX_PERCENTILE = 99.5
SERIES_COLORS = plt.cm.viridis(np.linspace(0.1, 0.85, len(CONFIGS)))


def label(name):
    return name.replace("_", " ")


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, filename), dpi=DPI)
    plt.close(fig)


def entry_stats(x):
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(np.median(x)),
        "skewness": float(stats.skew(x)),
        "kurtosis_excess": float(stats.kurtosis(x)),
    }


def fit_candidates(x):
    fits = {}
    for dist_name, dist in CANDIDATES.items():
        params = dist.fit(x)
        ks = stats.kstest(x, dist.name, args=params)
        loglik = float(np.sum(dist.logpdf(x, *params)))
        fits[dist_name] = {
            "params": [float(p) for p in params],
            "ks_stat": float(ks.statistic),
            "ks_p": float(ks.pvalue),
            "loglik": loglik,
            "aic": float(2 * len(params) - 2 * loglik),
        }
    return fits


np.random.seed(SEED)
mask = np.random.default_rng(SEED).random((N_RESERVOIR, N_RESERVOIR)) > SPARSITY

results = {}
matrices = {}
spectra = {}

for name, cfg in CONFIGS.items():
    print(f"Config {name}: {cfg}", flush=True)
    store = []

    def eigenvalue_fn(size, cfg=cfg, store=store):
        eigs = sample_eigenvalues_ginibre(size=size, **cfg)
        store.append(eigs)
        return eigs

    t0 = time.time()
    W_raw = generate_isospectral_sparse_matrix(
        eigenvalue_fn, N_RESERVOIR, SPARSITY, iterations=ITERATIONS, threshold=0.0, seed=SEED
    )
    elapsed = time.time() - t0

    W = W_raw.copy()
    W[np.abs(W) < THRESHOLD] = 0

    target = store[0]
    actual = np.linalg.eigvals(W)
    cost = np.abs(target[:, None] - actual[None, :])
    row, col = linear_sum_assignment(cost)
    matched = cost[row, col]

    nz = W != 0
    energy = float(np.sum(np.abs(W_raw) ** 2))
    entries = W[nz]

    matrices[name] = W
    spectra[name] = {"target": target, "actual": actual, "matched": matched, "row": row}

    results[name] = {
        "eigenvalue_params": cfg,
        "runtime_seconds": float(elapsed),
        "spectrum": {
            "mean_matched_distance": float(matched.mean()),
            "median_matched_distance": float(np.median(matched)),
            "max_matched_distance": float(matched.max()),
            "relative_matched_distance": float(matched.mean() / np.mean(np.abs(target))),
            "target_spectral_radius": float(np.max(np.abs(target))),
            "actual_spectral_radius": float(np.max(np.abs(actual))),
        },
        "mask": {
            "target_nonzero_fraction": float(mask.mean()),
            "achieved_nonzero_fraction": float(nz.mean()),
            "in_mask_kept_fraction": float(nz[mask].mean()),
            "off_mask_violation_fraction": float(nz[~mask].mean()),
            "off_mask_energy_fraction_raw": float(np.sum(np.abs(W_raw[~mask]) ** 2) / energy),
            "off_mask_energy_fraction_thresholded": float(np.sum(np.abs(W[~mask]) ** 2) / energy),
        },
        "entries": {
            "real": entry_stats(entries.real),
            "imag": entry_stats(entries.imag),
            "abs": entry_stats(np.abs(entries)),
        },
        "fits_real": fit_candidates(entries.real),
        "fits_imag": fit_candidates(entries.imag),
    }

config = {
    "method": "generate_isospectral_sparse_matrix + sample_eigenvalues_ginibre",
    "n_reservoir": N_RESERVOIR,
    "sparsity": SPARSITY,
    "iterations": ITERATIONS,
    "threshold": THRESHOLD,
    "seed": SEED,
    "note": "one matrix per config, identical sparsity mask across configs",
}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump({"config": config, "results": results}, f, indent=2)

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write(f"N={N_RESERVOIR}, sparsity={SPARSITY}, iterations={ITERATIONS}, threshold={THRESHOLD}, seed={SEED}\n")

    f.write("\n=== Spectral fidelity (Hungarian matching target vs actual) ===\n")
    for name, r in results.items():
        s = r["spectrum"]
        f.write(f"\n[{name}] params={r['eigenvalue_params']}, runtime={r['runtime_seconds']:.1f}s\n")
        f.write(f"  mean matched distance = {s['mean_matched_distance']:.5f} "
                f"(median {s['median_matched_distance']:.5f}, max {s['max_matched_distance']:.5f})\n")
        f.write(f"  relative to mean |lambda| = {s['relative_matched_distance']:.4f}\n")
        f.write(f"  spectral radius: target {s['target_spectral_radius']:.4f} -> "
                f"actual {s['actual_spectral_radius']:.4f}\n")

    f.write("\n=== Sparsity mask agreement ===\n")
    for name, r in results.items():
        m = r["mask"]
        f.write(f"\n[{name}]\n")
        f.write(f"  target nonzero fraction   = {m['target_nonzero_fraction']:.4f}\n")
        f.write(f"  achieved nonzero fraction = {m['achieved_nonzero_fraction']:.4f}\n")
        f.write(f"  in-mask entries kept      = {m['in_mask_kept_fraction']:.4f}\n")
        f.write(f"  off-mask entries nonzero  = {m['off_mask_violation_fraction']:.4f}\n")
        f.write(f"  off-mask energy (raw)     = {m['off_mask_energy_fraction_raw']:.6f}\n")
        f.write(f"  off-mask energy (thresh)  = {m['off_mask_energy_fraction_thresholded']:.6f}\n")

    f.write("\n=== Nonzero entry statistics ===\n")
    for name, r in results.items():
        f.write(f"\n[{name}]\n")
        for comp in ["real", "imag", "abs"]:
            s = r["entries"][comp]
            f.write(f"  {comp:>4}: n={s['n']}, mean={s['mean']:.4f}, std={s['std']:.4f}, "
                    f"min={s['min']:.4f}, max={s['max']:.4f}, median={s['median']:.4f}, "
                    f"skew={s['skewness']:.3f}, kurt={s['kurtosis_excess']:.3f}\n")

    for comp in ["real", "imag"]:
        f.write(f"\n=== MLE fits to {comp} parts of nonzero entries ===\n")
        for name, r in results.items():
            f.write(f"\n[{name}]\n")
            for dist_name, fr in sorted(r[f"fits_{comp}"].items(), key=lambda kv: kv[1]["aic"]):
                f.write(f"  {dist_name:>7}: AIC={fr['aic']:.0f}, loglik={fr['loglik']:.0f}, "
                        f"KS stat={fr['ks_stat']:.4f}, KS p={fr['ks_p']:.3g}, "
                        f"params={[round(p, 4) for p in fr['params']]}\n")

print("Plotting...", flush=True)

names = list(CONFIGS)
theta = np.linspace(0, 2 * np.pi, 500)
vmax = max(np.percentile(np.abs(matrices[n])[matrices[n] != 0], VMAX_PERCENTILE) for n in names)

for name in names:
    fig, ax = plt.subplots(figsize=SQUARE)
    im = ax.imshow(np.abs(matrices[name]), cmap=CMAP, vmin=0, vmax=vmax, interpolation="none")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    save(fig, f"heatmap_{name}.png")

for name in names:
    sp = spectra[name]
    fig, ax = plt.subplots(figsize=SQUARE)
    ax.plot(np.cos(theta), np.sin(theta), color=REFERENCE, linewidth=1)
    ax.scatter(sp["target"].real, sp["target"].imag, s=POINT_SIZE * 3, facecolors="none",
               edgecolors=SECONDARY, linewidths=0.8, label="target")
    ax.scatter(sp["actual"].real, sp["actual"].imag, s=POINT_SIZE, alpha=POINT_ALPHA,
               color=PRIMARY, label="actual")
    ax.set_aspect("equal")
    ax.set_xlabel("real")
    ax.set_ylabel("imaginary")
    ax.legend()
    save(fig, f"spectrum_{name}.png")

fig, ax = plt.subplots(figsize=WIDE)
ax.bar([label(n) for n in names], [results[n]["spectrum"]["mean_matched_distance"] for n in names],
       BAR_WIDTH * 2, color=PRIMARY)
ax.set_ylabel("mean eigenvalue distance")
save(fig, "spectral_distance.png")

fig, ax = plt.subplots(figsize=WIDE)
for name, color in zip(names, SERIES_COLORS):
    sp = spectra[name]
    ax.scatter(np.abs(sp["target"][sp["row"]]), sp["matched"], s=POINT_SIZE,
               alpha=POINT_ALPHA, color=color, label=label(name))
ax.set_yscale("log")
ax.set_xlabel("target modulus")
ax.set_ylabel("eigenvalue distance")
ax.legend()
save(fig, "spectral_error_vs_modulus.png")

fig, ax = plt.subplots(figsize=WIDE)
idx = np.arange(len(names))
ax.bar(idx - BAR_WIDTH / 2, [results[n]["mask"]["in_mask_kept_fraction"] for n in names],
       BAR_WIDTH, color=PRIMARY, label="kept inside mask")
ax.bar(idx + BAR_WIDTH / 2, [results[n]["mask"]["off_mask_violation_fraction"] for n in names],
       BAR_WIDTH, color=SECONDARY, label="nonzero outside mask")
ax.axhline(mask.mean(), color=REFERENCE, linewidth=1, linestyle="--",
           label=f"target density = {mask.mean():.3f}")
ax.set_xticks(idx)
ax.set_xticklabels([label(n) for n in names])
ax.set_ylabel("fraction of entries")
ax.legend()
save(fig, "mask_agreement.png")

styles = {"normal": "--", "cauchy": "-"}
axis_labels = {"real": "real part", "imag": "imaginary part"}
for comp in ["real", "imag"]:
    for name in names:
        entries = matrices[name][matrices[name] != 0]
        v = getattr(entries, comp)
        fig, ax = plt.subplots(figsize=WIDE)
        ax.hist(v, bins=100, density=True, color=PRIMARY, alpha=0.8, label="entries")
        grid = np.linspace(v.min(), v.max(), 500)
        for dist_name, dist in CANDIDATES.items():
            params = results[name][f"fits_{comp}"][dist_name]["params"]
            ax.plot(grid, dist.pdf(grid, *params), styles[dist_name], color=REFERENCE,
                    linewidth=1, label=dist_name)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, None)
        ax.set_xlabel(axis_labels[comp])
        ax.set_ylabel("density")
        ax.legend()
        save(fig, f"entries_{comp}_{name}.png")

print("Done", flush=True)
