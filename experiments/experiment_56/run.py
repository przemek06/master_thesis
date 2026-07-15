import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from generate import generate_isospectral_sparse_matrix
from distribution import sample_eigenvalues_ginibre

HERE = os.path.dirname(__file__)
N_RESERVOIR = 400
SPARSITY = 0.95
ITERATIONS = 50
N_MATRICES = 5
SEED = 0

CONFIGS = {
    "ring_narrow": {"r_min": 0.85, "r_max": 0.9, "alpha": 1.0},
    "ring_pushed": {"r_min": 0.1, "r_max": 0.9, "alpha": 8.0},
    "annulus": {"r_min": 0.3, "r_max": 0.9, "alpha": 2.0},
    "uniform_disk": {"r_min": 0.0, "r_max": 0.9, "alpha": 1.0},
}

CANDIDATES = {
    "normal": stats.norm,
    "laplace": stats.laplace,
    "cauchy": stats.cauchy,
    "student_t": stats.t,
}


def component_stats(x):
    n = len(x)
    ks_norm = stats.kstest(x, "norm", args=(x.mean(), x.std()))
    ks_unif = stats.kstest(x, "uniform", args=(x.min(), x.max() - x.min()))
    dagostino = stats.normaltest(x)
    return {
        "n": n,
        "mean": float(x.mean()),
        "variance": float(x.var()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(np.median(x)),
        "skewness": float(stats.skew(x)),
        "kurtosis_excess": float(stats.kurtosis(x)),
        "ks_vs_fitted_normal": {"stat": float(ks_norm.statistic), "p": float(ks_norm.pvalue)},
        "ks_vs_fitted_uniform": {"stat": float(ks_unif.statistic), "p": float(ks_unif.pvalue)},
        "dagostino_normality": {"stat": float(dagostino.statistic), "p": float(dagostino.pvalue)},
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
results = {}
pooled = {}
eig_samples = {}

for name, cfg in CONFIGS.items():
    print(f"Config {name}: {cfg}", flush=True)
    entries = []
    fractions_zero = []
    spectral_radii = []
    for i in range(N_MATRICES):
        W = generate_isospectral_sparse_matrix(
            lambda size: sample_eigenvalues_ginibre(size=size, **cfg),
            N_RESERVOIR,
            SPARSITY,
            iterations=ITERATIONS,
            seed=SEED + i,
        )
        nz = W[W != 0]
        entries.append(nz)
        fractions_zero.append(1.0 - len(nz) / W.size)
        spectral_radii.append(float(np.max(np.abs(np.linalg.eigvals(W)))))
        if i == 0:
            eig_samples[name] = np.linalg.eigvals(W)
    x = np.concatenate(entries)
    pooled[name] = x
    results[name] = {
        "fits_real": fit_candidates(x.real),
        "fits_imag": fit_candidates(x.imag),
        "eigenvalue_params": cfg,
        "n_matrices": N_MATRICES,
        "fraction_zero_mean": float(np.mean(fractions_zero)),
        "spectral_radius_mean": float(np.mean(spectral_radii)),
        "spectral_radius_std": float(np.std(spectral_radii)),
        "real": component_stats(x.real),
        "imag": component_stats(x.imag),
        "abs": component_stats(np.abs(x)),
    }

config = {
    "method": "generate_isospectral_sparse_matrix + sample_eigenvalues_ginibre",
    "n_reservoir": N_RESERVOIR,
    "sparsity": SPARSITY,
    "iterations": ITERATIONS,
    "n_matrices_per_config": N_MATRICES,
    "seed": SEED,
    "note": "stats computed on nonzero entries pooled over matrices",
}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump({"config": config, "results": results}, f, indent=2)

with open(os.path.join(HERE, "results.txt"), "w") as f:
    f.write("=== Statistics of nonzero reservoir entries (pooled over matrices) ===\n")
    for name, r in results.items():
        f.write(f"\n[{name}] params={r['eigenvalue_params']}, "
                f"fraction_zero={r['fraction_zero_mean']:.3f}, "
                f"spectral_radius={r['spectral_radius_mean']:.3f}±{r['spectral_radius_std']:.3f}\n")
        for comp in ["real", "imag", "abs"]:
            s = r[comp]
            f.write(f"  {comp:>4}: n={s['n']}, mean={s['mean']:.4f}, var={s['variance']:.5f}, "
                    f"min={s['min']:.4f}, max={s['max']:.4f}, median={s['median']:.4f}, "
                    f"skew={s['skewness']:.3f}, kurt={s['kurtosis_excess']:.3f}\n")
            f.write(f"        KS normal p={s['ks_vs_fitted_normal']['p']:.3g}, "
                    f"KS uniform p={s['ks_vs_fitted_uniform']['p']:.3g}, "
                    f"D'Agostino p={s['dagostino_normality']['p']:.3g}\n")
    for comp in ["real", "imag"]:
        f.write(f"\n=== MLE fits to {comp} parts of nonzero entries ===\n")
        for name, r in results.items():
            f.write(f"\n[{name}]\n")
            for dist_name, fr in sorted(r[f"fits_{comp}"].items(), key=lambda kv: kv[1]["aic"]):
                f.write(f"  {dist_name:>9}: AIC={fr['aic']:.0f}, loglik={fr['loglik']:.0f}, "
                        f"KS stat={fr['ks_stat']:.4f}, KS p={fr['ks_p']:.3g}, "
                        f"params={[round(p, 4) for p in fr['params']]}\n")

print("Plotting...", flush=True)

components = [("real", lambda x: x.real), ("imag", lambda x: x.imag), ("abs", np.abs)]
fig, axes = plt.subplots(len(CONFIGS), 3, figsize=(15, 4 * len(CONFIGS)))
for row, name in enumerate(CONFIGS):
    x_all = pooled[name]
    for col, (label, fn) in enumerate(components):
        ax = axes[row, col]
        x = fn(x_all)
        ax.hist(x, bins=60, density=True, color="darkorange", alpha=0.8)
        grid = np.linspace(x.min(), x.max(), 300)
        ax.plot(grid, stats.norm.pdf(grid, x.mean(), x.std()), color="black", linewidth=1.5, label="fitted normal")
        ax.axhline(1.0 / (x.max() - x.min()), color="steelblue", linewidth=1.5, linestyle="--", label="fitted uniform")
        ax.set_title(f"{name} — {label}")
        if row == 0 and col == 0:
            ax.legend()
fig.suptitle("Distribution of nonzero reservoir entries", y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(HERE, "distributions.png"), dpi=150)
plt.close()

theta = np.linspace(0, 2 * np.pi, 500)
for name in CONFIGS:
    eigs = eig_samples[name]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1)
    ax.scatter(eigs.real, eigs.imag, s=8, alpha=0.6, color="darkorange")
    ax.set_aspect("equal")
    ax.set_title(f"{name} — eigenvalues of generated W (first matrix)")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f"eigenvalues_{name}.png"), dpi=150)
    plt.close()

for name in CONFIGS:
    for comp in ["real", "imag"]:
        x = getattr(pooled[name], comp)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(x, bins=100, density=True, color="darkorange", alpha=0.8, label="entries")
        grid = np.linspace(x.min(), x.max(), 500)
        styles = {"normal": ":", "laplace": "--", "cauchy": "-", "student_t": "-."}
        for dist_name, dist in CANDIDATES.items():
            params = results[name][f"fits_{comp}"][dist_name]["params"]
            ax.plot(grid, dist.pdf(grid, *params), styles[dist_name], color="black", linewidth=1.5, label=dist_name)
        ax.set_yscale("log")
        ax.set_ylim(1e-3, None)
        ax.set_title(f"{name} — {comp} parts of nonzero entries vs fitted densities")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, f"fits_log_{comp}_{name}.png"), dpi=150)
        plt.close()

for name in CONFIGS:
    for comp in ["real", "imag"]:
        x = np.sort(getattr(pooled[name], comp))
        params = results[name][f"fits_{comp}"]["cauchy"]["params"]
        q = stats.cauchy.ppf((np.arange(1, len(x) + 1) - 0.5) / len(x), *params)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(q, x, s=3, alpha=0.5, color="darkorange")
        lims = [x.min(), x.max()]
        ax.plot(lims, lims, color="black", linewidth=1)
        ax.set_xlim(2 * lims[0], 2 * lims[1])
        ax.set_title(f"{name} — Q-Q plot vs fitted Cauchy ({comp} parts)")
        ax.set_xlabel("fitted Cauchy quantiles")
        ax.set_ylabel("ordered entries")
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, f"qq_cauchy_{comp}_{name}.png"), dpi=150)
        plt.close()

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, name in zip(axes.flat, CONFIGS):
    stats.probplot(pooled[name].real, dist="norm", plot=ax)
    ax.get_lines()[0].set(color="darkorange", markersize=3, alpha=0.5)
    ax.get_lines()[1].set(color="black")
    ax.set_title(f"{name} — real part vs normal")
fig.suptitle("Q-Q plots of real parts of nonzero entries")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "qq_real.png"), dpi=150)
plt.close()

print("Done.", flush=True)
