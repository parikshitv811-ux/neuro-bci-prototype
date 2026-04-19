"""
TSTA Dashboard — Full Validation Figure
=========================================
Generates an 8-panel figure:
  1. t-SNE of EEG direction embeddings
  2. Semantic direction compass (PCA 2D)
  3. Patch trajectories in latent space
  4. PLTA gate profiles
  5. Text embedding cosine heatmap
  6. Noise robustness curve
  7. Ablation bar chart
  8. Trajectory consistency per class

Saved to: outputs/figures/
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset

from tsta_project.config import FIGURES_DIR, TSTAConfig
from tsta_project.viz.tsne           import plot_tsne, PAL, CATS, BG, FG, GRID, _style
from tsta_project.viz.trajectory_plot import plot_direction_compass, plot_patch_trajectories
from tsta_project.viz.plta_viz        import plot_plta_gates, plot_text_cosine_heatmap


def _collect_embeddings(model, ds, cfg: TSTAConfig, device: str, subj: int):
    """Collect EEG directions, gates, and token trajectories for a subject."""
    mask = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long),
        ),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )
    model.eval()
    all_eeg, all_labels, all_gates, all_tokens = [], [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            tokens   = model.patcher(x)
            tokens   = model.transformer(tokens)
            ctx, gts = model.plta(tokens)
            direction = model.traj_head(tokens, ctx)
            direction = F.normalize(direction, dim=-1)
            all_eeg.append(direction.cpu())
            all_labels.append(y.cpu())
            all_gates.append(gts.cpu())
            all_tokens.append(tokens.cpu())

    return {
        "eeg":    torch.cat(all_eeg).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "gates":  torch.cat(all_gates)[0].numpy(),     # (G, N_patches)
        "tokens": torch.cat(all_tokens).numpy(),        # (N, N_patches, D)
    }


def build_dashboard(model,
                    ds,
                    cfg:             TSTAConfig,
                    device:          str,
                    ablation_results: dict = None,
                    noise_results:    dict = None,
                    subj:            int  = 1,
                    tag:             str  = "") -> str:
    """
    Build and save the full TSTA validation dashboard.

    Args:
        model:            Trained TSTA model
        ds:               EEGDataset
        cfg:              TSTAConfig
        device:           'cpu' | 'cuda'
        ablation_results: Output of run_ablation()
        noise_results:    Output of run_noise_robustness()
        subj:             Subject to visualize
        tag:              Filename tag (e.g., 'synthetic' or 'real')

    Returns:
        Path to saved figure
    """
    embs = _collect_embeddings(model, ds, cfg, device, subj)

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle(
        f"TSTA — Temporal Semantic Trajectory Alignment  |  {tag.upper() or 'Validation'} Results",
        color=FG, fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    # 1. t-SNE
    ax1 = fig.add_subplot(gs[0, :2])
    plot_tsne(embs["eeg"], embs["labels"], cfg.N_CLASSES, ax1)

    # 2. Direction compass
    ax2 = fig.add_subplot(gs[0, 2])
    plot_direction_compass(embs["eeg"], embs["labels"], cfg.N_CLASSES, ax2)

    # 3. Patch trajectories
    ax3 = fig.add_subplot(gs[0, 3])
    plot_patch_trajectories(embs["tokens"], embs["labels"], cfg.N_CLASSES, ax3)

    # 4. PLTA gates
    ax4 = fig.add_subplot(gs[1, :2])
    plot_plta_gates(embs["gates"], cfg, ax4)

    # 5. Text cosine heatmap
    ax5 = fig.add_subplot(gs[1, 2])
    plot_text_cosine_heatmap(model, cfg, device, ax5)

    # 6. Noise robustness
    ax6 = fig.add_subplot(gs[1, 3])
    _style(ax6)
    if noise_results:
        sigmas  = list(noise_results.keys())
        sdas_nr = list(noise_results.values())
        ax6.plot(sigmas, sdas_nr, color="#51cf66", lw=2, marker="o", markersize=5)
        ax6.fill_between(sigmas, 0, sdas_nr, alpha=0.15, color="#51cf66")
        ax6.axhline(0.3, color="#ffd43b", lw=1.2, linestyle="--", alpha=0.8,
                    label="Target SDAS=0.3")
        ax6.axhline(0.0, color="#ff6b6b", lw=0.8, linestyle=":", alpha=0.6,
                    label="Random")
        ax6.set_xlabel("Added noise σ", fontsize=8)
        ax6.set_ylabel("SDAS", fontsize=8)
        ax6.set_title("Noise robustness", fontsize=10)
        ax6.legend(fontsize=7, facecolor=GRID, edgecolor=GRID, labelcolor=FG)

    # 7. Ablation bar chart
    ax7 = fig.add_subplot(gs[2, :2])
    _style(ax7)
    if ablation_results:
        names  = list(ablation_results.keys())
        sdas_a = [ablation_results[n]["sdas"] for n in names]
        top1_a = [ablation_results[n]["top1_acc"] * 100 for n in names]
        x_pos  = np.arange(len(names))
        w      = 0.35
        b1 = ax7.bar(x_pos - w / 2, sdas_a, w,
                     color="#534AB7", alpha=0.85, label="SDAS", edgecolor=BG)
        ax7.bar(x_pos + w / 2, [t / 100 for t in top1_a], w,
                color="#1D9E75", alpha=0.85,
                label="Top-1 acc (normalized)", edgecolor=BG)
        ax7.axhline(0.4, color="#ffd43b", lw=1, linestyle="--", alpha=0.7,
                    label="SDAS target")
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(
            [n.replace("(", "").replace(")", "") for n in names],
            fontsize=7, color=FG, rotation=15, ha="right",
        )
        ax7.set_title("Ablation study", fontsize=10)
        ax7.set_ylabel("Score", fontsize=8)
        ax7.legend(fontsize=7, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
        for rect, val in zip(b1, sdas_a):
            ax7.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=7, color=FG,
            )

    # 8. Trajectory consistency
    ax8 = fig.add_subplot(gs[2, 2:])
    _style(ax8)
    cons_per_cls = []
    for cls in range(cfg.N_CLASSES):
        idx  = embs["labels"] == cls
        vecs = torch.tensor(embs["eeg"][idx])
        vecs = F.normalize(vecs, dim=-1)
        if len(vecs) > 1:
            cos_m = torch.matmul(vecs, vecs.T)
            off   = cos_m[~torch.eye(len(vecs), dtype=torch.bool)]
            cons_per_cls.append(float(off.mean()))
        else:
            cons_per_cls.append(0.0)
    bars = ax8.bar(
        range(cfg.N_CLASSES), cons_per_cls,
        color=[PAL[i % len(PAL)] for i in range(cfg.N_CLASSES)],
        alpha=0.85, edgecolor=BG,
    )
    ax8.set_xticks(range(cfg.N_CLASSES))
    ax8.set_xticklabels(
        [CATS[i] if i < len(CATS) else f"cls{i}" for i in range(cfg.N_CLASSES)],
        fontsize=8, color=FG, rotation=15,
    )
    ax8.axhline(0.5, color="#ffd43b", lw=1, linestyle="--", alpha=0.7,
                label="Consistency target")
    ax8.set_ylabel("Within-class cosine similarity", fontsize=8)
    ax8.set_title("Trajectory consistency per class", fontsize=10)
    ax8.legend(fontsize=7, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
    for bar, val in zip(bars, cons_per_cls):
        ax8.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=8, color=FG,
        )

    fname = f"tsta_dashboard_{tag or 'validation'}.png"
    out   = os.path.join(FIGURES_DIR, fname)
    plt.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  [Dashboard] Saved → {out}")
    return out
