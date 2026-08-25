import os
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "trajectory_candidates")
WARMUP = 1000
AFTER = 100

DPI = 150
WIDE = (7, 5)
REFERENCE = "black"
COLORS = {"customizable": "darkorange", "feedback": "steelblue"}

os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.png")):
    os.remove(f)

arrays = np.load(os.path.join(HERE, "arrays.npz"))

count = 0
for m in COLORS:
    target = arrays[f"all_target_{m}"][:, :, WARMUP:]
    pred = arrays[f"all_pred_{m}"][:, :, WARMUP:]
    steps = arrays[f"all_steps_{m}"]
    n_runs, n_series = target.shape[:2]
    for r in range(n_runs):
        for s in range(n_series):
            st = int(steps[r, s])
            end = min(st + AFTER, target.shape[2])
            x = np.arange(end)
            fig, ax = plt.subplots(figsize=WIDE)
            ax.plot(x, target[r, s, :end], color=REFERENCE, linewidth=1)
            ax.plot(x, pred[r, s, :end], color=COLORS[m], linestyle="--")
            ax.axvline(st, color=REFERENCE, linestyle="--", linewidth=1)
            ax.set_xlabel("step")
            ax.set_ylabel("value")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"predictions_{m}_run{r}_series{s}_steps{st:04d}.png"), dpi=DPI)
            plt.close(fig)
            count += 1

print(f"wrote {count} plots to {OUT}")
