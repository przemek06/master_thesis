import numpy as np
import matplotlib.pyplot as plt


def plot_circle_with_points(r: float, points: np.ndarray, filename: str):
    theta = np.linspace(0, 2 * np.pi, 500)
    circle_x = r * np.cos(theta)
    circle_y = r * np.sin(theta)

    fig, ax = plt.subplots()
    ax.plot(circle_x, circle_y, color="black", linewidth=1)
    ax.scatter(points.real, points.imag, s=10, alpha=0.6)
    ax.set_aspect("equal")
    plt.savefig(filename)
    plt.close(fig)


def plot_heatmap(matrices: list[np.ndarray], filename: str):
    combined = np.abs(sum(matrices))
    fig, ax = plt.subplots()
    im = ax.imshow(combined, cmap="hot")
    plt.colorbar(im, ax=ax)
    plt.savefig(filename)
    plt.close(fig)
