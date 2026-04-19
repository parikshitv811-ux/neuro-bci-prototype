"""
Phase 9 — Advanced Visualisation
==================================
Produces publication-quality figures proving deep trajectory properties:

  Fig 1 — Direction sphere (polar plot of class direction distributions)
  Fig 2 — Angular histogram per class
  Fig 3 — Trajectory curvature & smoothness comparison (correct vs incorrect)
  Fig 4 — Time-reversal direction shift comparison
  Fig 5 — Partial signal cos-sim decay curve
  Fig 6 — Domain shift SDAS bar chart
  Fig 7 — Amplitude invariance AIS curve
  Fig 8 — PCA 2D direction space scatter
  Fig 9 — Confusion matrix heatmap
  Fig 10 — Cross-subject alignment heatmap (CDAS per class)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

from tsta_project.config import FIGURES_DIR

# Class colour palette (consistent across all figs)
CLASS_COLORS = ["#58a6ff", "#3fb950", "#a371f7", "#f0883e", "#8b949e"]
DARK_BG      = "#0d1117"
CARD_BG      = "#161b22"
BORDER_CLR   = "#30363d"
TEXT_CLR     = "#e6edf3"
ACCENT_CLR   = "#58a6ff"


def _setup_fig(ncols: int = 1, nrows: int = 1,
               width: float = 6, height: float = 4.5):
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(width * ncols, height * nrows),
        facecolor=DARK_BG,
    )
    for ax in (np.array(axes).flatten() if hasattr(axes, "__iter__") else [axes]):
        ax.set_facecolor(CARD_BG)
        for spine in ax.spines.values():
            spine.set_color(BORDER_CLR)
        ax.tick_params(colors=TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(ACCENT_CLR)
    return fig, axes


def _save(fig, name: str) -> str:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight",
                facecolor=DARK_BG)
    plt.close(fig)
    return path


# ── Fig 1+2: PCA direction space scatter + angular histogram ─────────────────

def plot_direction_space(geo_results: dict, intents: list) -> str:
    pca_coords  = np.array(geo_results["pca_2d_coords"])
    labels      = np.array(geo_results["pca_2d_labels"])
    centroids   = geo_results["class_centroids_2d"]
    n_classes   = len(intents)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_color(BORDER_CLR)
        ax.tick_params(colors=TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(ACCENT_CLR)

    # Scatter
    for cls in range(n_classes):
        idx = labels == cls
        if idx.sum() == 0:
            continue
        ax1.scatter(pca_coords[idx, 0], pca_coords[idx, 1],
                    c=CLASS_COLORS[cls % len(CLASS_COLORS)],
                    alpha=0.45, s=18, label=intents[cls])
    # Centroids
    for cls_str, xy in centroids.items():
        cls = int(cls_str)
        ax1.scatter(*xy, c=CLASS_COLORS[cls % len(CLASS_COLORS)],
                    s=120, edgecolors="white", linewidths=1.2, zorder=5)
        ax1.annotate(intents[cls], xy, fontsize=7,
                     color=CLASS_COLORS[cls % len(CLASS_COLORS)],
                     ha="center", va="bottom", xytext=(0, 8),
                     textcoords="offset points")

    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.set_title(
        f"Direction Space — PCA 2D\n"
        f"ARI={geo_results['kmeans_ari']:.3f}  "
        f"ASS={geo_results['ass']:.3f}"
    )
    leg = ax1.legend(loc="upper right", fontsize=7,
                     facecolor=CARD_BG, edgecolor=BORDER_CLR)
    for t in leg.get_texts():
        t.set_color(TEXT_CLR)

    # Angular histogram (polar): angle of each direction vector in PCA space
    angles = np.arctan2(pca_coords[:, 1], pca_coords[:, 0])
    bins   = np.linspace(-np.pi, np.pi, 37)
    for cls in range(n_classes):
        idx = labels == cls
        if idx.sum() == 0:
            continue
        hist, _ = np.histogram(angles[idx], bins=bins)
        bin_centres = (bins[:-1] + bins[1:]) / 2
        ax2.bar(bin_centres, hist, width=(bins[1] - bins[0]) * 0.8,
                color=CLASS_COLORS[cls % len(CLASS_COLORS)],
                alpha=0.6, label=intents[cls])

    ax2.set_xlabel("Angle in PCA 2D (radians)")
    ax2.set_ylabel("Count")
    ax2.set_title("Angular Distribution per Class\n(PCA-projected direction vectors)")
    leg2 = ax2.legend(fontsize=7, facecolor=CARD_BG, edgecolor=BORDER_CLR)
    for t in leg2.get_texts():
        t.set_color(TEXT_CLR)

    fig.tight_layout(pad=1.5)
    return _save(fig, "adv_direction_space.png")


# ── Fig 3: Smoothness & curvature comparison ─────────────────────────────────

def plot_smoothness(smooth_results: dict, intents: list) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_color(BORDER_CLR)
        ax.tick_params(colors=TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(ACCENT_CLR)

    # Bar: per-class smoothness
    pc_smooth = smooth_results["per_class_smoothness"]
    classes   = sorted(pc_smooth.keys())
    vals      = [pc_smooth[c] for c in classes]
    xs        = np.arange(len(classes))
    bars      = ax1.bar(xs, vals, color=[CLASS_COLORS[c % 5] for c in classes], width=0.6)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([intents[c] for c in classes], rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Mean patch displacement norm")
    ax1.set_title("Temporal Smoothness per Class\n(lower = smoother trajectory)")

    # Correct vs incorrect
    sm_c  = smooth_results.get("smoothness_correct",   0)
    sm_i  = smooth_results.get("smoothness_incorrect", None)
    cu_c  = smooth_results.get("curvature_correct",    0)
    cu_i  = smooth_results.get("curvature_incorrect",  None)

    groups  = ["Correct"]
    s_vals  = [sm_c]
    c_vals  = [cu_c]
    colors  = ["#3fb950"]
    if sm_i is not None:
        groups.append("Incorrect")
        s_vals.append(sm_i)
        c_vals.append(cu_i if cu_i is not None else 0)
        colors.append("#f85149")

    x = np.arange(len(groups))
    w = 0.35
    ax2.bar(x - w/2, s_vals, width=w, color=colors,
            alpha=0.85, label="Smoothness")
    ax2.bar(x + w/2, c_vals, width=w, color=colors,
            alpha=0.5, label="Curvature")
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=9)
    ax2.set_ylabel("Trajectory metric")
    ax2.set_title("Trajectory Quality\nCorrect vs Incorrect Predictions")
    leg = ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER_CLR)
    for t in leg.get_texts():
        t.set_color(TEXT_CLR)

    fig.tight_layout(pad=1.5)
    return _save(fig, "adv_smoothness.png")


# ── Fig 4: Time reversal ─────────────────────────────────────────────────────

def plot_time_reversal(tr_results: dict, intents: list) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_color(BORDER_CLR)
        ax.tick_params(colors=TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(ACCENT_CLR)

    # Per-class TRE bar
    pc_tre  = tr_results.get("per_class_tre", {})
    classes = sorted(pc_tre.keys())
    vals    = [pc_tre[c] for c in classes]
    xs      = np.arange(len(classes))
    ax1.bar(xs, vals, color=[CLASS_COLORS[c % 5] for c in classes], width=0.6)
    ax1.axhline(y=0.3, color="#f0883e", linestyle="--", linewidth=1.2,
                label="TRE=0.3 threshold")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([intents[c] for c in classes], rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("Time Reversal Effect (TRE)")
    ax1.set_title(
        f"Time Reversal Effect per Class\n"
        f"Overall TRE = {tr_results['tre']:.4f}  "
        f"flip_pct = {tr_results['direction_flips_pct']*100:.1f}%"
    )
    leg = ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER_CLR)
    for t in leg.get_texts():
        t.set_color(TEXT_CLR)

    # Angular change gauge
    ang = tr_results["mean_angular_change"]
    theta = np.linspace(0, np.pi * 2, 300)
    circle_x, circle_y = np.cos(theta), np.sin(theta)
    ax2.plot(circle_x, circle_y, color=BORDER_CLR, lw=1)
    # Draw original dir (right) and rotated dir
    orig_angle  = 0.0
    rot_angle   = np.radians(ang)
    ax2.annotate("", xy=(np.cos(orig_angle), np.sin(orig_angle)),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>",
                                 color="#3fb950", lw=2))
    ax2.annotate("", xy=(np.cos(rot_angle + np.pi),
                          np.sin(rot_angle + np.pi)),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>",
                                 color="#f85149", lw=2))
    ax2.set_xlim(-1.4, 1.4); ax2.set_ylim(-1.4, 1.4)
    ax2.set_aspect("equal")
    ax2.set_title(f"Mean Angular Shift on Time Reversal\n{ang:.1f}°  (ideal = 180°)")
    ax2.text(0, -1.3, "Original dir →", ha="center", color="#3fb950", fontsize=8)
    ax2.text(0,  1.3, "← Reversed dir", ha="center", color="#f85149", fontsize=8)

    fig.tight_layout(pad=1.5)
    return _save(fig, "adv_time_reversal.png")


# ── Fig 5: Partial signal curve ──────────────────────────────────────────────

def plot_partial_signal(ps_results: dict) -> str:
    pf = ps_results["per_fraction"]
    fracs = sorted(float(k) for k in pf.keys())
    cos_vals  = [pf[str(f)]["cos_sim"] for f in fracs]
    sdas_vals = [pf[str(f)]["sdas"]    for f in fracs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_color(BORDER_CLR)
        ax.tick_params(colors=TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(ACCENT_CLR)

    fracs_pct = [f * 100 for f in fracs]
    ax1.plot(fracs_pct, cos_vals, "-o", color=ACCENT_CLR, linewidth=2,
             markersize=8, label="cos-sim(partial, full)")
    ax1.fill_between(fracs_pct, cos_vals, alpha=0.15, color=ACCENT_CLR)
    ax1.axhline(y=0.5, color="#f0883e", linestyle="--", linewidth=1.2,
                label="0.5 threshold")
    ax1.set_xlabel("Signal fraction used (%)")
    ax1.set_ylabel("Cosine similarity to full-signal direction")
    ax1.set_title("Early Intent Detectability\n(direction similarity vs truncation)")
    ax1.set_ylim(-0.1, 1.1)
    leg = ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER_CLR)
    for t in leg.get_texts():
        t.set_color(TEXT_CLR)

    ax2.plot(fracs_pct, sdas_vals, "-s", color="#3fb950", linewidth=2,
             markersize=8, label="SDAS")
    ax2.axhline(y=0.4, color="#f0883e", linestyle="--", linewidth=1.2,
                label="Target SDAS=0.4")
    ax2.fill_between(fracs_pct, sdas_vals, alpha=0.15, color="#3fb950")
    ax2.set_xlabel("Signal fraction used (%)")
    ax2.set_ylabel("SDAS")
    ax2.set_title("SDAS vs Signal Fraction")
    leg2 = ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER_CLR)
    for t in leg2.get_texts():
        t.set_color(TEXT_CLR)

    fig.tight_layout(pad=1.5)
    return _save(fig, "adv_partial_signal.png")


# ── Fig 6: Domain shift bar chart ────────────────────────────────────────────

def plot_domain_shift(ds_results: dict) -> str:
    baseline = ds_results["baseline_sdas"]
    corrs    = ds_results["per_corruption"]
    names    = list(corrs.keys())
    vals     = [corrs[n]["sdas"] for n in names]
    degs     = [corrs[n]["degradation_pct"] for n in names]

    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_color(BORDER_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(ACCENT_CLR)

    xs = np.arange(len(names))
    colors = [("#3fb950" if v >= 0.4 * 0.5 else "#f85149") for v in vals]
    bars = ax.bar(xs, vals, color=colors, width=0.6, alpha=0.85)
    ax.axhline(y=baseline, color=ACCENT_CLR, linestyle="--",
               linewidth=1.5, label=f"Baseline SDAS={baseline:.3f}")
    ax.axhline(y=0.4 * 0.5, color="#f0883e", linestyle=":",
               linewidth=1.2, label="Half-target threshold")
    ax.set_xticks(xs)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("SDAS")
    ax.set_title("Domain Shift Robustness\n(SDAS under distribution shift)")
    for bar_, deg in zip(bars, degs):
        ax.text(bar_.get_x() + bar_.get_width() / 2,
                bar_.get_height() + 0.01,
                f"{deg:+.1f}%", ha="center", va="bottom",
                color=TEXT_CLR, fontsize=8)
    leg = ax.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER_CLR)
    for t in leg.get_texts():
        t.set_color(TEXT_CLR)

    fig.tight_layout(pad=1.5)
    return _save(fig, "adv_domain_shift.png")


# ── Fig 7: Amplitude invariance curve ────────────────────────────────────────

def plot_amplitude_invariance(amp_results: dict) -> str:
    pa   = amp_results["per_scale_ais"]
    psdas = amp_results["per_scale_sdas"]
    scales = sorted(pa.keys())
    ais   = [pa[s]    for s in scales]
    sdas  = [psdas[s] for s in scales]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(CARD_BG)
        for sp in ax.spines.values():
            sp.set_color(BORDER_CLR)
        ax.tick_params(colors=TEXT_CLR)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.title.set_color(ACCENT_CLR)

    ax1.plot(scales, ais, "-o", color=ACCENT_CLR, linewidth=2, markersize=8)
    ax1.fill_between(scales, ais, alpha=0.15, color=ACCENT_CLR)
    ax1.axhline(y=0.85, color="#f0883e", linestyle="--",
                label="AIS=0.85 threshold")
    ax1.axvline(x=1.0,  color="#8b949e", linestyle=":", linewidth=1, label="Scale=1x")
    ax1.set_xlabel("Amplitude scale factor")
    ax1.set_ylabel("AIS (cosine similarity)")
    ax1.set_title(
        f"Amplitude Invariance Score\nRandom-scale AIS = {amp_results['ais_random']:.4f}"
    )
    ax1.set_ylim(-0.1, 1.1)
    leg = ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER_CLR)
    for t in leg.get_texts():
        t.set_color(TEXT_CLR)

    ax2.plot(scales, sdas, "-s", color="#3fb950", linewidth=2, markersize=8)
    ax2.fill_between(scales, sdas, alpha=0.15, color="#3fb950")
    ax2.axhline(y=0.4, color="#f0883e", linestyle="--",
                label="SDAS target=0.4")
    ax2.axvline(x=1.0, color="#8b949e", linestyle=":", linewidth=1)
    ax2.set_xlabel("Amplitude scale factor")
    ax2.set_ylabel("SDAS")
    ax2.set_title("SDAS vs Amplitude Scale")
    ax2.set_xscale("log")
    leg2 = ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER_CLR)
    for t in leg2.get_texts():
        t.set_color(TEXT_CLR)

    fig.tight_layout(pad=1.5)
    return _save(fig, "adv_amplitude_invariance.png")


# ── Fig 8: Cross-subject alignment heatmap ───────────────────────────────────

def plot_cross_subject_alignment(inv_results: dict, intents: list) -> str:
    pc_aln  = inv_results["per_class_alignment"]
    classes = sorted(pc_aln.keys())
    vals    = [pc_aln[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_color(BORDER_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(ACCENT_CLR)

    xs    = np.arange(len(classes))
    colors_bar = [CLASS_COLORS[c % 5] for c in classes]
    ax.bar(xs, vals, color=colors_bar, width=0.6, alpha=0.85)
    ax.axhline(y=0.0, color="#8b949e", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([intents[c] for c in classes], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean cross-subject cosine similarity")
    ax.set_title(
        f"Cross-Subject Direction Alignment\n"
        f"CDAS = {inv_results['cdas']:+.4f}  "
        f"same={inv_results['same_class_mean_cos']:+.4f}  "
        f"diff={inv_results['diff_class_mean_cos']:+.4f}"
    )
    fig.tight_layout(pad=1.5)
    return _save(fig, "adv_direction_invariance.png")


# ── Fig 9: Confusion matrix ───────────────────────────────────────────────────

def plot_confusion_matrix(fm_results: dict, intents: list) -> str:
    cm = np.array(fm_results["confusion_matrix"])
    n  = cm.shape[0]
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=DARK_BG)
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_color(BORDER_CLR)
    ax.tick_params(colors=TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(ACCENT_CLR)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=TEXT_CLR)

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else TEXT_CLR
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=9, color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([intents[i][:6] for i in range(n)], rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels([intents[i][:6] for i in range(n)], fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (normalised)\nAccuracy={fm_results['accuracy']*100:.1f}%")

    fig.tight_layout(pad=1.5)
    return _save(fig, "adv_confusion_matrix.png")


# ── Master: build all advanced figures ───────────────────────────────────────

def build_advanced_figures(results: dict, intents: list) -> list[str]:
    """
    Build all advanced visualisation figures.

    Args:
        results: dict with keys matching analysis module names
        intents: list of class intent labels

    Returns:
        List of saved figure paths.
    """
    saved = []
    try:
        if "geometric_structure" in results:
            saved.append(plot_direction_space(results["geometric_structure"], intents))
        if "temporal_smoothness" in results:
            saved.append(plot_smoothness(results["temporal_smoothness"], intents))
        if "time_reversal" in results:
            saved.append(plot_time_reversal(results["time_reversal"], intents))
        if "partial_signal" in results:
            saved.append(plot_partial_signal(results["partial_signal"]))
        if "domain_shift" in results:
            saved.append(plot_domain_shift(results["domain_shift"]))
        if "amplitude_invariance" in results:
            saved.append(plot_amplitude_invariance(results["amplitude_invariance"]))
        if "direction_invariance" in results:
            saved.append(plot_cross_subject_alignment(
                results["direction_invariance"], intents
            ))
        if "failure_modes" in results:
            saved.append(plot_confusion_matrix(results["failure_modes"], intents))
    except Exception as e:
        print(f"  [Viz warning] {e}")

    print(f"\n  [Viz] Saved {len(saved)} advanced figures to {FIGURES_DIR}/")
    for p in saved:
        print(f"    → {os.path.basename(p)}")
    return saved
