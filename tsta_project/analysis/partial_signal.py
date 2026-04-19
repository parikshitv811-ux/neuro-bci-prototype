"""
Phase 7 — Partial Signal Robustness (Early Intent Detectability)
=================================================================
Claim:  EEG direction encodes semantic intent early in the epoch.
        Even using only the first 25% of the signal should produce
        directions well-aligned with the full-signal direction.

Method:
  • For fractions f ∈ {0.1, 0.25, 0.50, 0.75, 1.00}:
      - Keep first f * T samples; zero-pad the rest.
      - Encode truncated signal → dir_f.
  • Compare dir_f with dir_full using:
      - cosine similarity (direction similarity)
      - SDAS on truncated input

Metric:  "Early intent detectability" — cosine sim between truncated and full direction.
         Also shows at what fraction SDAS first crosses 0.4 target.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tsta_project.config          import TSTAConfig
from tsta_project.training.metrics import compute_sdas


def run_partial_signal(model,
                       ds,
                       cfg:      TSTAConfig,
                       device:   str,
                       subj:     int   = 1,
                       fractions: list = None) -> dict:
    """
    Args:
        model:     Trained TSTA model
        ds:        EEGDataset
        cfg:       TSTAConfig
        device:    'cpu' | 'cuda'
        subj:      Subject ID
        fractions: List of time fractions to test

    Returns:
        {
          'per_fraction': {frac: {'cos_sim': float, 'sdas': float}},
          'first_target_fraction': float | None,
          'claim_supported': bool,
        }
    """
    print("\n" + "=" * 60)
    print("  PHASE 7 — Partial Signal Robustness (Early Intent)")
    print("=" * 60)

    if fractions is None:
        fractions = [0.10, 0.25, 0.50, 0.75, 1.00]

    mask = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]
    T = X_s.shape[-1]

    # Full signal directions (reference)
    model.eval()
    full_ds = TensorDataset(
        torch.tensor(X_s, dtype=torch.float32),
        torch.tensor(y_s, dtype=torch.long),
    )
    full_l = DataLoader(full_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    full_dirs, full_labels = [], []
    with torch.no_grad():
        for x_b, y_b in full_l:
            d, _, _ = model(x_b.to(device), y_b.to(device))
            full_dirs.append(d.cpu())
            full_labels.append(y_b)
    full_dirs   = F.normalize(torch.cat(full_dirs),   dim=-1)
    full_labels = torch.cat(full_labels).numpy()

    per_fraction = {}
    print(f"\n  {'Fraction':>10}  {'T used':>8}  {'cos-sim':>10}  {'SDAS':>8}  Bar")
    print(f"  {'─'*58}")

    first_target = None
    for frac in fractions:
        n_keep = max(1, int(T * frac))
        X_tr   = X_s.copy()
        X_tr[:, :, n_keep:] = 0.0   # zero-pad the future

        tr_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_s,  dtype=torch.long),
        )
        tr_l  = DataLoader(tr_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        # Directions from truncated signal
        tr_dirs = []
        with torch.no_grad():
            for x_b, y_b in tr_l:
                d, _, _ = model(x_b.to(device), y_b.to(device))
                tr_dirs.append(d.cpu())
        tr_dirs = F.normalize(torch.cat(tr_dirs), dim=-1)

        cos_sim = float(F.cosine_similarity(full_dirs, tr_dirs, dim=-1).mean())
        sdas_m  = compute_sdas(model, tr_l, device, cfg.N_CLASSES)["sdas"]

        per_fraction[frac] = {
            "cos_sim": round(cos_sim, 4),
            "sdas":    round(sdas_m, 4),
            "n_samples_used": n_keep,
        }

        if first_target is None and sdas_m >= cfg.TARGET_WITHIN_SDAS:
            first_target = frac

        bar  = "█" * max(0, int(cos_sim * 20))
        sdas_flag = "✓" if sdas_m >= cfg.TARGET_WITHIN_SDAS else " "
        print(f"  {frac:>10.0%}  {n_keep:>8d}  {cos_sim:>10.4f}  "
              f"{sdas_m:>7.4f} {sdas_flag} {bar}")

    if first_target is not None:
        print(f"\n  First fraction meeting SDAS target: {first_target:.0%}")
    else:
        print(f"\n  SDAS target not reached with any partial fraction tested.")

    # Early detectability: cos-sim at 25% should be > 0.6
    cos_25 = per_fraction.get(0.25, {}).get("cos_sim", 0.0)
    supported = cos_25 > 0.5
    print(f"  cos-sim at 25% fraction : {cos_25:.4f}")
    print(f"  Claim supported: {'✓ YES' if supported else '✗ NO'}"
          f"  (cos-sim > 0.5 at 25% signal)")

    return {
        "per_fraction":          {str(k): v for k, v in per_fraction.items()},
        "first_target_fraction": first_target,
        "claim_supported":       supported,
    }
