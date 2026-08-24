import os
import glob
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "trajectory_candidates")
N_EXAMPLES = 30

DPI = 150
WIDE = (7, 5)
REFERENCE = "black"
COLORS = {"customizable": "darkorange", "feedback": "steelblue"}

os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.png")):
    os.remove(f)

arrays = np.load(os.path.join(HERE, "arrays.npz"))
target = arrays["test_target"]
horizon = target.shape[1]
lead = np.arange(1, horizon + 1)
examples = np.linspace(0, len(target) - 1, N_EXAMPLES).astype(int)

for m in COLORS:
    pred = arrays[f"test_pred_{m}"]
    for i in examples:
        fig, ax = plt.subplots(figsize=WIDE)
        ax.plot(lead, target[i], color=REFERENCE, linewidth=1)
        ax.plot(lead, pred[i], color=COLORS[m], linestyle="--")
        ax.set_xlabel("hours ahead")
        ax.set_ylabel("standardized temperature")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"predictions_{m}_ex{i:05d}.png"), dpi=DPI)
        plt.close(fig)

print(f"wrote {len(examples) * len(COLORS)} plots to {OUT}")
