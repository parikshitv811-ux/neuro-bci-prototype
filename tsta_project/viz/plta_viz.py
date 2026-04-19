"""
PLTA Gate and Text Embedding Visualizations
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def plot_plta_gates(gates: np.ndarray, cfg, ax):
    """Plot learned PLTA gate profiles over time."""
    _style(ax)
    n_patches = gates.shape[1]
    patch_centers_s = np.array([
        (i * cfg.PATCH_STEP + cfg.PATCH_LEN / 2) / cfg.SFREQ
        for i in range(n_patches)
    ])
    gate_colors = ["#00d4ff", "#ff9f7f", "#a8e6cf"]
    gate_labels = ["P2 gate (200ms)", "N2/P3 gate (300ms)", "Late gate (500ms)"]

    for g in range(gates.shape[0]):
        if g >= len(gate_colors):
            break
        ax.plot(patch_centers_s, gates[g], color=gate_colors[g],
                lw=2.0, label=gate_labels[g], alpha=0.9)
        ax.fill_between(patch_centers_s, 0, gates[g],
                        color=gate_colors[g], alpha=0.15)

    for t_ms in [200, 300, 500]:
        ax.axvline(t_ms / 1000, color=GRID, lw=0.8, linestyle="--", alpha=0.6)
        ymax = gates.max() if gates.max() > 0 else 0.1
        ax.text(t_ms / 1000, ymax * 1.02, f"{t_ms}ms",
                color=FG, fontsize=7, ha="center")

    ax.set_title("PLTA — learned temporal gate profiles", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("Gate weight", fontsize=8)
    ax.legend(fontsize=8, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
    epoch_len = cfg.N_SAMPLES / cfg.SFREQ
    ax.set_xlim(0, epoch_len)


def plot_text_cosine_heatmap(model, cfg, device: str, ax):
    """Cosine similarity heatmap of all class text embeddings."""
    _style(ax)
    ids = torch.arange(cfg.N_CLASSES, device=device)
    with torch.no_grad():
        model.eval()
        te = model.encode_text(ids).cpu()

    import torch.nn.functional as F
    cos_mat = torch.matmul(te, te.T).numpy()

    im = ax.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(cfg.N_CLASSES))
    ax.set_yticks(range(cfg.N_CLASSES))
    short = [c[:4] for c in CATS[:cfg.N_CLASSES]]
    ax.set_xticklabels(short, fontsize=7, color=FG, rotation=30)
    ax.set_yticklabels(short, fontsize=7, color=FG)
    for i in range(cfg.N_CLASSES):
        for j in range(cfg.N_CLASSES):
            ax.text(
                j, i,
                f"{cos_mat[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(cos_mat[i, j]) > 0.5 else FG,
            )
    ax.set_title("Text embedding\ncosine similarity", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
