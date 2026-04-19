"""
Phase 6 — Time Reversal Test (Causality Check)
================================================
Claim:  Trajectory direction encodes temporal causality.
        Reversing the EEG signal in time should flip or substantially
        change the direction vector — proving the model uses temporal order.

Method:
  • For each sample, encode with original signal → dir_orig.
  • Flip time axis: x_rev = x.flip(-1).
  • Encode reversed signal → dir_rev.
  • Measure:
      cos_sim(dir_orig, dir_rev)        — near −1 means direction flips
      angular_change_deg                 — how many degrees the direction rotates
      reversal_effect = 1 − mean(cos_sim(orig, rev))   — 0=no change, 2=perfect flip

Metric:  "Time reversal effect" (TRE)
         TRE = 1 − mean_cos_sim(dir_orig, dir_rev)
         TRE ≈ 2.0 → perfect flip (ideal causal model)
         TRE ≈ 0.0 → direction invariant to time order (bad — not causal)
         TRE ≈ 1.0 → partial effect (realistic for deep models)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tsta_project.config import TSTAConfig


def run_time_reversal(model,
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
          'mean_cos_sim':         float,  # orig vs reversed (neg = flip)
          'mean_angular_change':  float,  # degrees
          'tre':                  float,  # time reversal effect
          'per_class_tre':        {cls: float},
          'direction_flips_pct':  float,  # % samples where cos < 0
          'claim_supported':      bool,
        }
    """
    print("\n" + "=" * 60)
    print("  PHASE 6 — Time Reversal Test (Causality)")
    print("=" * 60)

    mask   = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]

    model.eval()
    orig_dirs, rev_dirs, all_labels = [], [], []

    with torch.no_grad():
        ds_ = TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long),
        )
        dl = DataLoader(ds_, batch_size=cfg.BATCH_SIZE, shuffle=False)

        for x_b, y_b in dl:
            x_b = x_b.to(device)
            y_b = y_b.to(device)
            d_orig, _, _ = model(x_b,          y_b)
            d_rev,  _, _ = model(x_b.flip(-1), y_b)   # reverse time axis
            orig_dirs.append(d_orig.cpu())
            rev_dirs.append(d_rev.cpu())
            all_labels.append(y_b.cpu())

    orig_dirs  = F.normalize(torch.cat(orig_dirs),  dim=-1)
    rev_dirs   = F.normalize(torch.cat(rev_dirs),   dim=-1)
    all_labels = torch.cat(all_labels).numpy()

    cos_sims = F.cosine_similarity(orig_dirs, rev_dirs, dim=-1)  # (N,)
    mean_cos = float(cos_sims.mean())
    ang_deg  = float(torch.acos(cos_sims.clamp(-1, 1)).mean().item() * 180.0 / np.pi)
    tre      = 1.0 - mean_cos
    flip_pct = float((cos_sims < 0).float().mean())

    # Per-class TRE
    intents = cfg.INTENTS if hasattr(cfg, "INTENTS") else [str(c) for c in range(cfg.N_CLASSES)]
    per_class_tre = {}
    print(f"\n  {'Class':<18}  {'cos(orig,rev)':>14}  {'TRE':>8}  {'Δ deg':>8}")
    print(f"  {'─'*54}")
    for cls in range(cfg.N_CLASSES):
        idx = all_labels == cls
        if idx.sum() < 2:
            continue
        cos_c = float(cos_sims[idx].mean())
        ang_c = float(torch.acos(cos_sims[idx].clamp(-1, 1)).mean().item() * 180 / np.pi)
        tre_c = 1.0 - cos_c
        per_class_tre[cls] = round(tre_c, 4)
        print(f"  {intents[cls]:<18}  {cos_c:>+14.4f}  {tre_c:>8.4f}  {ang_c:>7.1f}°")

    print(f"\n  Overall mean cos(orig, rev) : {mean_cos:+.4f}")
    print(f"  Mean angular change         : {ang_deg:.1f}°")
    print(f"  Time Reversal Effect (TRE)  : {tre:.4f}")
    print(f"  Samples with direction flip : {flip_pct*100:.1f}%")

    # TRE > 0.3 shows meaningful temporal causality
    supported = tre > 0.3
    print(f"\n  Claim supported: {'✓ YES' if supported else '✗ NO'}"
          f"  (TRE > 0.3 threshold)")

    return {
        "mean_cos_sim":        round(mean_cos, 4),
        "mean_angular_change": round(ang_deg,  2),
        "tre":                 round(tre,       4),
        "per_class_tre":       per_class_tre,
        "direction_flips_pct": round(flip_pct, 4),
        "claim_supported":     supported,
    }
