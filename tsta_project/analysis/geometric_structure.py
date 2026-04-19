"""
Phase 5 — Geometric Structure of Direction Space
==================================================
Claim:  EEG direction vectors occupy consistent angular regions per class,
        forming a structured latent geometry — not random scatter.

Method:
  • Collect all direction vectors from a trained model.
  • Compute:
      - Intra-class variance  (mean ||d − mean_d||² per class)
      - Inter-class angle     (mean pairwise angular distance between class centres)
      - Angular separability score (ASS)
      - k-means cluster quality (purity, silhouette-like score)
  • PCA projection onto 2D for visualisation.

Metric:  "Angular Separability Score" (ASS)
         ASS = (inter-class centroid angle) / (mean intra-class spread)
         Higher → classes form tighter, better-separated cones.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from tsta_project.config import TSTAConfig


def run_geometric_structure(model,
                            ds,
                            cfg:    TSTAConfig,
                            device: str,
                            subj:   int = 1) -> dict:
    """
    Args:
        model:  Trained TSTA model
        ds:     EEGDataset
        cfg:    TSTAConfig
        device: 'cpu' | 'cuda'
        subj:   Subject to analyse

    Returns:
        {
          'ass':                      float,   # Angular separability score
          'intra_class_variance':     {cls: float},
          'inter_class_angle_deg':    float,
          'kmeans_ari':               float,   # Adjusted Rand Index vs true labels
          'kmeans_purity':            float,
          'pca_2d_coords':            list,    # [(x, y)] for each sample
          'pca_2d_labels':            list,
          'class_centroids_2d':       list,
          'claim_supported':          bool,
        }
    """
    print("\n" + "=" * 60)
    print("  PHASE 5 — Geometric Structure of Direction Space")
    print("=" * 60)

    mask   = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]

    # ── Collect direction vectors ────────────────────────────────────────────
    model.eval()
    all_dirs, all_labels = [], []
    full_ds = TensorDataset(
        torch.tensor(X_s, dtype=torch.float32),
        torch.tensor(y_s, dtype=torch.long),
    )
    full_l = DataLoader(full_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for x_b, y_b in full_l:
            d, _, _ = model(x_b.to(device), y_b.to(device))
            all_dirs.append(d.cpu())
            all_labels.append(y_b)

    all_dirs   = F.normalize(torch.cat(all_dirs),   dim=-1).numpy()
    all_labels = torch.cat(all_labels).numpy()
    n_classes  = cfg.N_CLASSES
    intents    = cfg.INTENTS if hasattr(cfg, "INTENTS") else [str(c) for c in range(n_classes)]

    # ── Class centroids ──────────────────────────────────────────────────────
    centroids    = {}
    intra_var    = {}
    for cls in range(n_classes):
        idx = all_labels == cls
        if idx.sum() < 2:
            continue
        vecs   = all_dirs[idx]                              # (N_c, D)
        mean_v = vecs.mean(axis=0)
        mean_v = mean_v / (np.linalg.norm(mean_v) + 1e-8)  # unit centroid
        centroids[cls] = mean_v
        # Intra-class variance = mean squared distance from centroid (on sphere)
        cos_sim        = (vecs @ mean_v)                    # (N_c,)
        ang_spread     = np.arccos(np.clip(cos_sim, -1, 1)) # radians
        intra_var[cls] = float(np.var(ang_spread))

    # ── Inter-class angle ────────────────────────────────────────────────────
    inter_angles = []
    cls_list = sorted(centroids.keys())
    for i, c1 in enumerate(cls_list):
        for j, c2 in enumerate(cls_list):
            if j <= i:
                continue
            cos  = float(np.clip(centroids[c1] @ centroids[c2], -1, 1))
            ang  = float(np.degrees(np.arccos(cos)))
            inter_angles.append(ang)

    mean_inter_angle = float(np.mean(inter_angles)) if inter_angles else 0.0
    mean_intra_spread = float(np.mean(list(intra_var.values()))) if intra_var else 1e-8
    ass = mean_inter_angle / (mean_intra_spread * 10000 + 1e-8)

    print(f"\n  {'Class':<18}  {'N':>5}  {'Intra-var (rad²)':>18}")
    print(f"  {'─'*46}")
    for cls in cls_list:
        idx = all_labels == cls
        print(f"  {intents[cls]:<18}  {int(idx.sum()):>5}  {intra_var.get(cls, float('nan')):>18.6f}")
    print(f"\n  Mean inter-class angle : {mean_inter_angle:.2f}°")
    print(f"  Mean intra-class spread: {mean_intra_spread:.6f} rad²")
    print(f"  Angular Separability   : {ass:.4f}")

    # ── k-Means clustering ───────────────────────────────────────────────────
    km    = KMeans(n_clusters=n_classes, n_init=10, random_state=42)
    preds = km.fit_predict(all_dirs)
    ari   = float(adjusted_rand_score(all_labels, preds))

    # Cluster purity
    purity_sum = 0
    for k in range(n_classes):
        mask_k = preds == k
        if mask_k.sum() == 0:
            continue
        mode_count = np.bincount(all_labels[mask_k]).max()
        purity_sum += mode_count
    purity = float(purity_sum / len(all_labels))

    print(f"\n  k-Means ARI (vs true labels) : {ari:.4f}")
    print(f"  k-Means Purity               : {purity:.4f}")

    # ── PCA 2D projection ─────────────────────────────────────────────────────
    pca     = PCA(n_components=2, random_state=42)
    coords  = pca.fit_transform(all_dirs)          # (N, 2)
    var_exp = pca.explained_variance_ratio_
    print(f"\n  PCA: 2D variance explained = {var_exp[0]*100:.1f}% + {var_exp[1]*100:.1f}%")

    centroids_2d = {}
    for cls in cls_list:
        idx = all_labels == cls
        if idx.sum() < 1:
            continue
        centroids_2d[cls] = coords[idx].mean(axis=0).tolist()

    supported = (ari > 0.3) and (mean_inter_angle > 10.0)
    print(f"\n  Claim supported: {'✓ YES' if supported else '✗ NO'}"
          f"  (ARI > 0.3 and inter-class angle > 10°)")

    return {
        "ass":                      round(ass, 4),
        "intra_class_variance":     {k: round(v, 6) for k, v in intra_var.items()},
        "inter_class_angle_deg":    round(mean_inter_angle, 2),
        "kmeans_ari":               round(ari, 4),
        "kmeans_purity":            round(purity, 4),
        "pca_explained_var":        [round(float(v), 4) for v in var_exp],
        "pca_2d_coords":            coords.tolist(),
        "pca_2d_labels":            all_labels.tolist(),
        "class_centroids_2d":       centroids_2d,
        "claim_supported":          supported,
    }
