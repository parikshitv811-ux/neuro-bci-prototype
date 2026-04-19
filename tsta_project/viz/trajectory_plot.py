"""
PCA Trajectory and Direction Compass Visualizations
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PAL  = ["#1D9E75", "#534AB7", "#D85A30", "#BA7517", "#888780"]
CATS = ["communication", "navigation", "action", "selection", "idle"]
BG   = "#0f0f1a"
FG   = "#e0e0e0"
GRID = "#2a2a3a"


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def plot_direction_compass(eeg: np.ndarray, labels: np.ndarray,
                           n_classes: int, ax):
    """Direction compass: mean class direction vector in PCA 2D space."""
    _style(ax)
    pca  = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(eeg)

    for cls in range(n_classes):
        idx = labels == cls
        if not idx.any():
            continue
        mean_dir = proj[idx].mean(axis=0)
        norm = np.linalg.norm(mean_dir) + 1e-8
        mean_dir /= norm
        ax.annotate(
            "",
            xy=mean_dir,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=PAL[cls % len(PAL)], lw=2.0),
        )
        ax.text(
            mean_dir[0] * 1.15,
            mean_dir[1] * 1.15,
            CATS[cls] if cls < len(CATS) else f"cls{cls}",
            color=PAL[cls % len(PAL)],
            fontsize=7,
            ha="center",
        )

    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.axhline(0, color=GRID, lw=0.5)
    ax.axvline(0, color=GRID, lw=0.5)
    ax.set_title("Semantic direction compass\n(PCA 2D)", fontsize=10)
    ax.set_aspect("equal")


def plot_patch_trajectories(tokens: np.ndarray, labels: np.ndarray,
                            n_classes: int, ax):
    """
    PCA trajectory plot: show how each class's mean token path
    moves through latent space from patch 0 → last patch.
    ○ = start, ★ = end.
    """
    _style(ax)
    N, T, D = tokens.shape
    pca = PCA(n_components=2, random_state=42)
    pca.fit(tokens.reshape(-1, D))

    for cls in range(n_classes):
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        mean_traj = tokens[idx].mean(axis=0)   # (T, D)
        tproj     = pca.transform(mean_traj)    # (T, 2)
        color = PAL[cls % len(PAL)]
        ax.plot(tproj[:, 0], tproj[:, 1], color=color, lw=1.5,
                label=CATS[cls] if cls < len(CATS) else f"cls{cls}",
                alpha=0.9)
        ax.scatter(tproj[0,  0], tproj[0,  1],  c=color, s=40, marker="o", zorder=5)
        ax.scatter(tproj[-1, 0], tproj[-1, 1], c=color, s=40, marker="*", zorder=5)

    ax.set_title("Trajectories in latent space\n(○=start  ★=end)", fontsize=10)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.legend(fontsize=7, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
