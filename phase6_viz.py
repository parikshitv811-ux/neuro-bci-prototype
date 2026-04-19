"""
TSTA Phase 6 — Visualizations
==============================
1. t-SNE of EEG direction embeddings
2. Trajectory plots (time → latent space, PCA)
3. Direction vector compass per class
4. Antonym separation heatmap
5. PLTA gate profiles (learned temporal attention)
6. Noise robustness curve
7. Ablation bar chart
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

from phase3_model import TSTA, TSTAConfig

# Palette
PAL  = ['#1D9E75','#534AB7','#D85A30','#BA7517','#888780']
CATS = ['communication','navigation','action','selection','idle']
BG   = '#0f0f1a'
FG   = '#e0e0e0'
GRID = '#2a2a3a'


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.xaxis.label.set_color(FG); ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def collect_embeddings(model, ds, cfg, device, subj=1):
    mask = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]
    loader = DataLoader(
        TensorDataset(torch.tensor(X_s).float(), torch.tensor(y_s).long()),
        batch_size=cfg.BATCH_SIZE, shuffle=False
    )
    model.eval()
    all_eeg, all_labels, all_gates = [], [], []
    all_tokens = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # Collect patch tokens for trajectory plots
            tokens   = model.patcher(x)
            tokens   = model.transformer(tokens)
            ctx, gts = model.plta(tokens)
            direction= model.traj_head(tokens, ctx)
            direction= F.normalize(direction, dim=-1)
            all_eeg.append(direction.cpu())
            all_labels.append(y.cpu())
            all_gates.append(gts.cpu())
            all_tokens.append(tokens.cpu())
    return {
        'eeg':    torch.cat(all_eeg).numpy(),
        'labels': torch.cat(all_labels).numpy(),
        'gates':  torch.cat(all_gates)[0].numpy(),   # (G, N_patches)
        'tokens': torch.cat(all_tokens).numpy(),      # (N, N_patches, D)
    }


def build_figure(model, ds, cfg, device, ablation_results, noise_results, subj=1):
    embs = collect_embeddings(model, ds, cfg, device, subj)

    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.suptitle('TSTA — Temporal Semantic Trajectory Alignment  |  Validation Results',
                 color=FG, fontsize=14, fontweight='bold', y=0.98)
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.4)

    # ── 1. t-SNE ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _style(ax1)
    tsne = TSNE(n_components=2, perplexity=min(30, len(embs['labels'])//5),
                random_state=42, max_iter=500)
    proj = tsne.fit_transform(embs['eeg'])
    for cls in range(cfg.N_CLASSES):
        idx = embs['labels'] == cls
        ax1.scatter(proj[idx, 0], proj[idx, 1], c=PAL[cls], label=CATS[cls],
                   s=18, alpha=0.7, edgecolors='none')
    ax1.set_title('t-SNE of EEG direction embeddings', fontsize=10)
    ax1.legend(fontsize=8, facecolor=GRID, edgecolor=GRID, labelcolor=FG,
              markerscale=1.5, ncol=2)
    ax1.set_xlabel('t-SNE 1', fontsize=8); ax1.set_ylabel('t-SNE 2', fontsize=8)

    # ── 2. Direction compass (PCA 2D) ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2)
    pca  = PCA(n_components=2, random_state=42)
    proj2 = pca.fit_transform(embs['eeg'])
    for cls in range(cfg.N_CLASSES):
        idx = embs['labels'] == cls
        mean_dir = proj2[idx].mean(axis=0)
        mean_dir /= (np.linalg.norm(mean_dir) + 1e-8)
        ax2.annotate('', xy=mean_dir, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=PAL[cls], lw=2.0))
        ax2.text(mean_dir[0] * 1.1, mean_dir[1] * 1.1, CATS[cls],
                color=PAL[cls], fontsize=7, ha='center')
    ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4)
    ax2.axhline(0, color=GRID, lw=0.5); ax2.axvline(0, color=GRID, lw=0.5)
    ax2.set_title('Semantic direction compass\n(PCA 2D)', fontsize=10)
    ax2.set_aspect('equal')

    # ── 3. Trajectory in PCA space (per class, time→) ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 3])
    _style(ax3)
    pca3 = PCA(n_components=2, random_state=42)
    # Flatten tokens for PCA fit: (N*T, D)
    N, T, D = embs['tokens'].shape
    flat    = embs['tokens'].reshape(-1, D)
    pca3.fit(flat)
    for cls in range(cfg.N_CLASSES):
        idx = np.where(embs['labels'] == cls)[0]
        if len(idx) == 0: continue
        # Average token trajectory across samples of this class
        mean_traj = embs['tokens'][idx].mean(axis=0)  # (T, D)
        tproj     = pca3.transform(mean_traj)         # (T, 2)
        ax3.plot(tproj[:, 0], tproj[:, 1], color=PAL[cls], lw=1.5,
                label=CATS[cls], alpha=0.9)
        ax3.scatter(tproj[0,  0], tproj[0,  1], c=PAL[cls], s=40, marker='o', zorder=5)
        ax3.scatter(tproj[-1, 0], tproj[-1, 1], c=PAL[cls], s=40, marker='*', zorder=5)
    ax3.set_title('Trajectories in latent space\n(○=start  ★=end)', fontsize=10)
    ax3.set_xlabel('PC1', fontsize=8); ax3.set_ylabel('PC2', fontsize=8)
    ax3.legend(fontsize=7, facecolor=GRID, edgecolor=GRID, labelcolor=FG)

    # ── 4. PLTA gate profiles ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    _style(ax4)
    gates     = embs['gates']    # (G, N_patches)
    n_patches = gates.shape[1]
    patch_centers_s = np.array(
        [(i * cfg.PATCH_STEP + cfg.PATCH_LEN / 2) / cfg.SFREQ
         for i in range(n_patches)])
    gate_colors = ['#00d4ff', '#ff9f7f', '#a8e6cf']
    gate_labels = ['P2 gate (200ms)', 'N2/P3 gate (300ms)', 'Late gate (500ms)']
    for g in range(gates.shape[0]):
        ax4.plot(patch_centers_s, gates[g], color=gate_colors[g],
                lw=2.0, label=gate_labels[g], alpha=0.9)
        ax4.fill_between(patch_centers_s, 0, gates[g],
                        color=gate_colors[g], alpha=0.15)
    for t_ms in [200, 300, 500]:
        ax4.axvline(t_ms/1000, color=GRID, lw=0.8, linestyle='--', alpha=0.6)
        ax4.text(t_ms/1000, ax4.get_ylim()[1] if ax4.get_ylim()[1] > 0 else 0.1,
                f'{t_ms}ms', color=FG, fontsize=7, ha='center')
    ax4.set_title('PLTA — learned temporal gate profiles', fontsize=10)
    ax4.set_xlabel('Time (s)', fontsize=8)
    ax4.set_ylabel('Gate weight', fontsize=8)
    ax4.legend(fontsize=8, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
    ax4.set_xlim(0, cfg.EPOCH_LEN if hasattr(cfg,'EPOCH_LEN') else 2.0)

    # ── 5. Text embedding cosine heatmap ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    _style(ax5)
    ids = torch.arange(cfg.N_CLASSES)
    with torch.no_grad():
        model.eval()
        te = model.encode_text(ids.to(next(model.parameters()).device)).cpu()
    cos_mat = torch.matmul(te, te.T).numpy()
    im = ax5.imshow(cos_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax5.set_xticks(range(cfg.N_CLASSES))
    ax5.set_yticks(range(cfg.N_CLASSES))
    short = ['comm','nav','act','sel','idle']
    ax5.set_xticklabels(short, fontsize=7, color=FG, rotation=30)
    ax5.set_yticklabels(short, fontsize=7, color=FG)
    for i in range(cfg.N_CLASSES):
        for j in range(cfg.N_CLASSES):
            ax5.text(j, i, f'{cos_mat[i,j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(cos_mat[i,j]) > 0.5 else FG)
    ax5.set_title('Text embedding\ncosine similarity', fontsize=10)
    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    # ── 6. Noise robustness curve ──────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 3])
    _style(ax6)
    if noise_results:
        sigmas  = list(noise_results.keys())
        sdas_nr = list(noise_results.values())
        ax6.plot(sigmas, sdas_nr, color='#51cf66', lw=2, marker='o', markersize=5)
        ax6.fill_between(sigmas, 0, sdas_nr, alpha=0.15, color='#51cf66')
        ax6.axhline(0.3, color='#ffd43b', lw=1.2, linestyle='--', alpha=0.8,
                   label='Target SDAS=0.3')
        ax6.axhline(0.0, color='#ff6b6b', lw=0.8, linestyle=':', alpha=0.6,
                   label='Random')
        ax6.set_xlabel('Added noise σ', fontsize=8)
        ax6.set_ylabel('SDAS', fontsize=8)
        ax6.set_title('Noise robustness', fontsize=10)
        ax6.legend(fontsize=7, facecolor=GRID, edgecolor=GRID, labelcolor=FG)

    # ── 7. Ablation bar chart ──────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, :2])
    _style(ax7)
    if ablation_results:
        names = list(ablation_results.keys())
        sdas_a = [ablation_results[n]['sdas'] for n in names]
        top1_a = [ablation_results[n]['top1_acc'] * 100 for n in names]
        x_pos  = np.arange(len(names))
        w      = 0.35
        b1 = ax7.bar(x_pos - w/2, sdas_a, w, color='#534AB7', alpha=0.85,
                    label='SDAS', edgecolor=BG)
        b2 = ax7.bar(x_pos + w/2, [t/100 for t in top1_a], w, color='#1D9E75',
                    alpha=0.85, label='Top-1 acc (normalized)', edgecolor=BG)
        ax7.axhline(0.4, color='#ffd43b', lw=1, linestyle='--', alpha=0.7,
                   label='SDAS target')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels([n.replace('(','').replace(')','') for n in names],
                           fontsize=7, color=FG, rotation=15, ha='right')
        ax7.set_title('Ablation study', fontsize=10)
        ax7.set_ylabel('Score', fontsize=8)
        ax7.legend(fontsize=7, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
        for rect, val in zip(b1, sdas_a):
            ax7.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.01,
                    f'{val:.3f}', ha='center', fontsize=7, color=FG)

    # ── 8. Trajectory consistency per class ───────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 2:])
    _style(ax8)
    # Within-class cosine similarity (trajectory consistency)
    cons_per_cls = []
    for cls in range(cfg.N_CLASSES):
        idx  = embs['labels'] == cls
        vecs = torch.tensor(embs['eeg'][idx])
        vecs = F.normalize(vecs, dim=-1)
        if len(vecs) > 1:
            cos_m = torch.matmul(vecs, vecs.T)
            off   = cos_m[~torch.eye(len(vecs), dtype=torch.bool)]
            cons_per_cls.append(float(off.mean()))
        else:
            cons_per_cls.append(0)
    bars = ax8.bar(range(cfg.N_CLASSES), cons_per_cls, color=PAL, alpha=0.85,
                  edgecolor=BG)
    ax8.set_xticks(range(cfg.N_CLASSES))
    ax8.set_xticklabels(CATS[:cfg.N_CLASSES], fontsize=8, color=FG, rotation=15)
    ax8.axhline(0.5, color='#ffd43b', lw=1, linestyle='--', alpha=0.7,
               label='Consistency target')
    ax8.set_ylabel('Within-class cosine similarity', fontsize=8)
    ax8.set_title('Trajectory consistency per class', fontsize=10)
    ax8.legend(fontsize=7, facecolor=GRID, edgecolor=GRID, labelcolor=FG)
    for bar, val in zip(bars, cons_per_cls):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=8, color=FG)

    out = '/home/claude/tsta/tsta_validation.png'
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  [Viz] Saved → {out}")
    return out
