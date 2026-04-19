"""
Phase 2 — Amplitude Invariance Test
=====================================
Claim:  Trajectory encodes semantic meaning, NOT signal amplitude.
        Scaling EEG amplitude should NOT change the direction vector.

Method:
  • Take a trained model and all data for one subject.
  • For each sample, compute direction with original signal.
  • Re-scale the signal by a random factor in [0.5, 2.0].
  • Recompute direction.
  • Measure cosine similarity between original and scaled directions.

Metric:  "Amplitude invariance score" (AIS)
         AIS = mean cosine_similarity(dir_original, dir_scaled)

Expected: AIS close to 1.0 — direction is scale-invariant.
Also tests SDAS preservation under scaling.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tsta_project.config  import TSTAConfig
from tsta_project.training.metrics import compute_sdas


def run_amplitude_invariance(model,
                             ds,
                             cfg:    TSTAConfig,
                             device: str,
                             subj:   int  = 1,
                             scales: list = None,
                             n_random: int = 200) -> dict:
    """
    Args:
        model:    Trained TSTA model
        ds:       EEGDataset
        cfg:      TSTAConfig
        device:   'cpu' | 'cuda'
        subj:     Subject ID to test
        scales:   Explicit scale factors to test (besides random)
        n_random: Number of random scale factors drawn from [0.5, 2.0]

    Returns:
        {
          'ais_random':      float,   # mean cos-sim across n_random scales
          'per_scale_sdas':  {scale: sdas},
          'per_scale_ais':   {scale: ais},
          'baseline_sdas':   float,
          'claim_supported': bool,
        }
    """
    print("\n" + "=" * 60)
    print("  PHASE 2 — Amplitude Invariance Test")
    print("=" * 60)

    mask    = ds.subjects == subj
    X_s     = ds.X[mask]
    y_s     = ds.y[mask]
    full_ds = TensorDataset(
        torch.tensor(X_s, dtype=torch.float32),
        torch.tensor(y_s, dtype=torch.long),
    )
    full_l  = DataLoader(full_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # ── Baseline directions ──────────────────────────────────────────────────
    model.eval()
    orig_dirs, orig_labels = [], []
    with torch.no_grad():
        for x_b, y_b in full_l:
            d, _, _ = model(x_b.to(device), y_b.to(device))
            orig_dirs.append(d.cpu())
            orig_labels.append(y_b)
    orig_dirs   = torch.cat(orig_dirs)      # (N, D)
    orig_labels = torch.cat(orig_labels)

    baseline_m  = compute_sdas(model, full_l, device, cfg.N_CLASSES)
    baseline_sdas = baseline_m["sdas"]
    print(f"\n  Baseline SDAS : {baseline_sdas:.4f}")

    # ── Fixed scale factor sweep ─────────────────────────────────────────────
    if scales is None:
        scales = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]

    per_scale_sdas: dict[float, float] = {}
    per_scale_ais:  dict[float, float] = {}

    print(f"\n  {'Scale':>6}  {'AIS':>8}  {'SDAS':>8}  Bar")
    print(f"  {'─'*48}")

    for sc in scales:
        X_sc  = (torch.tensor(X_s, dtype=torch.float32) * sc).to(device)
        dirs_sc = []
        with torch.no_grad():
            for i in range(0, len(X_sc), cfg.BATCH_SIZE):
                x_b = X_sc[i: i + cfg.BATCH_SIZE]
                y_b = torch.tensor(y_s[i: i + cfg.BATCH_SIZE], dtype=torch.long).to(device)
                d, _, _ = model(x_b, y_b)
                dirs_sc.append(d.cpu())
        dirs_sc = torch.cat(dirs_sc)

        cos_sim = F.cosine_similarity(orig_dirs, dirs_sc, dim=-1).mean().item()
        per_scale_ais[sc] = round(cos_sim, 4)

        # SDAS with scaled input
        sc_ds = TensorDataset(X_sc.cpu(), torch.tensor(y_s, dtype=torch.long))
        sc_l  = DataLoader(sc_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        sdas_sc = compute_sdas(model, sc_l, device, cfg.N_CLASSES)["sdas"]
        per_scale_sdas[sc] = sdas_sc

        bar = "█" * max(0, int(cos_sim * 20))
        print(f"  {sc:>6.2f}x  {cos_sim:>8.4f}  {sdas_sc:>8.4f}  {bar}")

    # ── Random scale factor test ─────────────────────────────────────────────
    rng = np.random.RandomState(42)
    rand_scales = rng.uniform(0.5, 2.0, size=n_random)
    ais_list = []
    for sc in rand_scales:
        X_sc = torch.tensor(X_s * sc, dtype=torch.float32)
        dirs_sc = []
        with torch.no_grad():
            for i in range(0, len(X_sc), cfg.BATCH_SIZE):
                x_b = X_sc[i: i + cfg.BATCH_SIZE].to(device)
                y_b = torch.tensor(y_s[i: i + cfg.BATCH_SIZE], dtype=torch.long).to(device)
                d, _, _ = model(x_b, y_b)
                dirs_sc.append(d.cpu())
        dirs_sc = torch.cat(dirs_sc)
        ais_list.append(
            F.cosine_similarity(orig_dirs, dirs_sc, dim=-1).mean().item()
        )

    ais_random = float(np.mean(ais_list))
    print(f"\n  Random-scale AIS (n={n_random}, scales∈[0.5,2.0]): {ais_random:.4f}")

    supported = ais_random > 0.85
    print(f"  Claim supported: {'✓ YES' if supported else '✗ NO'}"
          f"  (AIS > 0.85 threshold)")

    return {
        "ais_random":      round(ais_random, 4),
        "per_scale_sdas":  {round(k, 2): v for k, v in per_scale_sdas.items()},
        "per_scale_ais":   {round(k, 2): v for k, v in per_scale_ais.items()},
        "baseline_sdas":   round(baseline_sdas, 4),
        "claim_supported": supported,
    }
