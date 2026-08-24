import os
import json
import numpy as np
from scipy import stats

HERE = os.path.dirname(__file__)
PATH = os.path.join(HERE, "results.json")

results = json.load(open(PATH))
summary = results["summary"]

feedback = np.array(summary["feedback"]["run_means"])
customizable = np.array(summary["customizable"]["run_means"])

t_stat, p_value = stats.ttest_ind(feedback, customizable, equal_var=False)
better = "customizable" if customizable.mean() < feedback.mean() else "feedback"

summary["statistical_test"] = {
    "type": "welch_t_test",
    "note": "unpaired: both models see identical data in every run, so the samples differ only by search seed",
    "t_statistic": float(t_stat),
    "p_value": float(p_value),
    "significant": bool(p_value < 0.05),
    "better_model": better,
}

with open(PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Welch t-test: t={t_stat:.3f}, p={p_value:.4f}, better={better}")
