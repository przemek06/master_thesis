import os
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "trajectory_candidates")
WARMUP = 200
START = 500
WINDOW = 200

DPI = 150
WIDE = (7, 5)
REFERENCE = "black"
COLORS = {"customizable": "darkorange", "feedback": "steelblue"}

os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.png")):
    os.remove(f)

arrays = np.load(os.path.join(HERE, "arrays.npz"))
seg = slice(START, START + WINDOW)
t = np.arange(WARMUP + START, WARMUP + START + WINDOW)

count = 0
for m in COLORS:
    target = arrays[f"all_target_{m}"]
    pred = arrays[f"all_pred_{m}"]
    n_runs, n_series = target.shape[:2]
    for r in range(n_runs):
        for s in range(n_series):
            fig, ax = plt.subplots(figsize=WIDE)
            ax.plot(t, target[r, s, seg], color=REFERENCE, linewidth=1)
            ax.plot(t, pred[r, s, seg], color=COLORS[m], linestyle="--")
            ax.set_xlabel("step")
            ax.set_ylabel("output")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, f"predictions_{m}_run{r}_series{s}.png"), dpi=DPI)
            plt.close(fig)
            count += 1

print(f"wrote {count} plots to {OUT}")
