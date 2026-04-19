"""
Phase 1 — Direction Invariance Test (Cross-Subject Geometry)
=============================================================
Claim:  The same semantic class produces aligned direction vectors
        across different subjects, even without any cross-subject
        training signal.

Method:
  • Encode all samples with each subject's own trained model.
  • Compute mean unit direction per class per subject.
  • Measure pairwise cosine similarity of mean directions across subjects
    for same-class pairs vs different-class pairs.

Metric:  "Cross-subject direction alignment score"
         CDAS = mean(same-class cross-subj cos) − mean(diff-class cross-subj cos)

Expected: CDAS > 0  (same class aligns across subjects)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tsta_project.config import TSTAConfig


def _get_directions(model, X: np.ndarray, y: np.ndarray,
                    cfg: TSTAConfig, device: str) -> tuple:
    """Return (eeg_dirs, labels) tensors for the full dataset X."""
    ds  = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    dl  = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    dirs, labels = [], []
    model.eval()
    with torch.no_grad():
        for x_b, y_b in dl:
            d, _, _ = model(x_b.to(device), y_b.to(device))
            dirs.append(d.cpu())
            labels.append(y_b)
    return torch.cat(dirs), torch.cat(labels)


def run_direction_invariance(models: dict,
                             ds,
                             cfg:    TSTAConfig,
                             device: str) -> dict:
    """
    Args:
        models:  {subj_id: trained TSTA model}
        ds:      EEGDataset
        cfg:     TSTAConfig
        device:  'cpu' | 'cuda'

    Returns:
        {
          'cdas':                  float,   # primary metric
          'same_class_mean_cos':   float,
          'diff_class_mean_cos':   float,
          'per_class_alignment':   {cls_id: float},
          'subject_mean_dirs':     {subj_id: {cls_id: np.ndarray}},
          'claim_supported':       bool,
        }
    """
    print("\n" + "=" * 60)
    print("  PHASE 1 — Cross-Subject Direction Invariance")
    print("=" * 60)

    subj_ids  = sorted(models.keys())
    n_classes = cfg.N_CLASSES

    # ── Compute mean direction per (subject, class) ──────────────────────────
    mean_dirs: dict[int, dict[int, np.ndarray]] = {}
    for sid in subj_ids:
        model = models[sid]
        mask  = ds.subjects == sid
        X_s, y_s = ds.X[mask], ds.y[mask]
        dirs, labels = _get_directions(model, X_s, y_s, cfg, device)

        mean_dirs[sid] = {}
        for cls in range(n_classes):
            idx = (labels == cls)
            if idx.sum() < 1:
                continue
            vecs = F.normalize(dirs[idx], dim=-1)
            mean_dirs[sid][cls] = F.normalize(
                vecs.mean(dim=0, keepdim=True), dim=-1
            ).squeeze(0).numpy()

    # ── Pairwise cross-subject cosine by class ───────────────────────────────
    same_cos, diff_cos = [], []
    per_class: dict[int, list] = {c: [] for c in range(n_classes)}

    for i, s1 in enumerate(subj_ids):
        for j, s2 in enumerate(subj_ids):
            if j <= i:
                continue
            for c1 in range(n_classes):
                if c1 not in mean_dirs[s1] or c1 not in mean_dirs[s2]:
                    continue
                v1 = torch.tensor(mean_dirs[s1][c1])
                v2 = torch.tensor(mean_dirs[s2][c1])
                cos_same = float(F.cosine_similarity(v1.unsqueeze(0),
                                                     v2.unsqueeze(0)))
                same_cos.append(cos_same)
                per_class[c1].append(cos_same)

                # different-class pairs
                for c2 in range(n_classes):
                    if c2 == c1:
                        continue
                    if c2 not in mean_dirs[s2]:
                        continue
                    v3 = torch.tensor(mean_dirs[s2][c2])
                    diff_cos.append(
                        float(F.cosine_similarity(v1.unsqueeze(0),
                                                  v3.unsqueeze(0)))
                    )

    same_mean = float(np.mean(same_cos))  if same_cos else 0.0
    diff_mean = float(np.mean(diff_cos))  if diff_cos else 0.0
    cdas      = same_mean - diff_mean
    per_class_aln = {cls: float(np.mean(v)) for cls, v in per_class.items() if v}

    print(f"\n  Same-class cross-subject cosine : {same_mean:+.4f}")
    print(f"  Diff-class cross-subject cosine : {diff_mean:+.4f}")
    print(f"  CDAS (δ)                        : {cdas:+.4f}")
    print(f"\n  Per-class alignment:")
    intents = cfg.INTENTS if hasattr(cfg, "INTENTS") else [str(c) for c in range(n_classes)]
    for c, v in per_class_aln.items():
        bar  = "█" * max(0, int((v + 1) * 10))
        print(f"    {intents[c]:15s} {v:+.4f}  {bar}")

    supported = cdas > 0.05
    print(f"\n  Claim supported: {'✓ YES' if supported else '✗ NO'}"
          f"  (CDAS > 0.05 threshold)")

    return {
        "cdas":                 round(cdas,      4),
        "same_class_mean_cos":  round(same_mean, 4),
        "diff_class_mean_cos":  round(diff_mean, 4),
        "per_class_alignment":  {k: round(v, 4) for k, v in per_class_aln.items()},
        "subject_mean_dirs":    {s: {c: v.tolist() for c, v in d.items()}
                                 for s, d in mean_dirs.items()},
        "claim_supported":      supported,
    }
