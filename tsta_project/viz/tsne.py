"""
t-SNE Visualization of EEG Direction Embeddings
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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


def plot_tsne(eeg: np.ndarray, labels: np.ndarray, n_classes: int, ax):
    """Plot 2D t-SNE of EEG direction embeddings."""
    _style(ax)
    perp = min(30, max(5, len(labels) // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=500)
    proj = tsne.fit_transform(eeg)
    for cls in range(n_classes):
        idx = labels == cls
        if not idx.any():
            continue
        ax.scatter(
            proj[idx, 0], proj[idx, 1],
            c=PAL[cls % len(PAL)],
            label=CATS[cls] if cls < len(CATS) else f"cls{cls}",
            s=18, alpha=0.7, edgecolors="none",
        )
    ax.set_title("t-SNE of EEG direction embeddings", fontsize=10)
    ax.legend(fontsize=8, facecolor=GRID, edgecolor=GRID,
              labelcolor=FG, markerscale=1.5, ncol=2)
    ax.set_xlabel("t-SNE 1", fontsize=8)
    ax.set_ylabel("t-SNE 2", fontsize=8)
