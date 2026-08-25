import os
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "delay_candidates")
N_DELAYS = 30

DPI = 150
SQUARE = (6, 6)
REFERENCE = "black"
POINT_SIZE = 8
POINT_ALPHA = 0.6
COLORS = {"customizable": "darkorange", "feedback": "steelblue"}

os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.png")):
    os.remove(f)

arrays = np.load(os.path.join(HERE, "arrays.npz"))
k_max = arrays["target_customizable"].shape[1]
delays = np.linspace(0, k_max - 1, N_DELAYS).astype(int)

for m in COLORS:
    target = arrays[f"target_{m}"]
    pred = arrays[f"pred_{m}"]
    for d in delays:
        t = target[:, d]
        p = pred[:, d]
        fig, ax = plt.subplots(figsize=SQUARE)
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], color=REFERENCE, linewidth=1)
        ax.scatter(t, p, s=POINT_SIZE, alpha=POINT_ALPHA, color=COLORS[m])
        ax.set_aspect("equal")
        ax.set_xlabel("target")
        ax.set_ylabel("prediction")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"target_vs_prediction_{m}_delay{d + 1:03d}.png"), dpi=DPI)
        plt.close(fig)

print(f"wrote {len(delays) * len(COLORS)} plots to {OUT}")
