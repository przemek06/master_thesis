## Rules
1. Don't write comments
2. Don't write excessive prints
3. Keep the code simple and clean
4. Don't include unnecessary exception handling and edge cases
5. Don't write code that is not directly related to the task at hand

## Plots
1. One plot per image, never a grid of subplots
2. No titles, no suptitles, no legends
3. Axis labels are short, lowercase, full words, no underscores and no symbols like `Re` or `lambda`
4. Config and series names are rendered with spaces instead of underscores; file names keep underscores
5. `dpi=150`, `tight_layout()` before saving, close the figure afterwards
6. Figure size `(6, 6)` for heatmaps and complex plane scatters, `(7, 5)` for bars, histograms and 1-D scatters
7. Heatmaps use `cmap="viridis"`, `vmin=0`, `interpolation="none"`, no ticks, a colorbar with `fraction=0.046, pad=0.02`, and a `vmax` shared across every image being compared
8. First series is `darkorange`, second is `steelblue`, reference lines and fitted curves are `black` with `linewidth=1`; series are distinguished by color alone, never by a legend label
9. More than two series use `plt.cm.viridis` sampled over `0.1` to `0.85`
10. Scatter points use `s=8, alpha=0.6`; grouped bars use width `0.4` placed at `idx -/+ 0.2`
11. Values that would otherwise go in a title or legend belong in `results.txt`

## Context
This is a research project to explore alternative ways of constructing and optimizing echo state networks. 