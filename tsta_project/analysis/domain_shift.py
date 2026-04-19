"""
Phase 3 — Domain Shift Simulation
====================================
Claim:  Trajectory direction remains meaningful under distribution shift.

Simulated shifts applied to the test set:
  A. Noise boost      — triple Gaussian noise level
  B. Frequency shift  — add 3 Hz carrier offset to all oscillations
  C. Channel dropout  — randomly zero 20% of channels per sample
  D. Combined         — all three simultaneously

Method:
  • Train on original synthetic data (subject 1).
  • Apply each corruption to the test data.
  • Evaluate SDAS on corrupted inputs.
  • Compare to clean baseline.

Metric:  "Domain transfer SDAS"
         DT-SDAS degradation = (baseline_sdas − shifted_sdas) / baseline_sdas
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tsta_project.config          import TSTAConfig
from tsta_project.training.metrics import compute_sdas


# ── Corruption functions ─────────────────────────────────────────────────────

def _noise_boost(X: np.ndarray, sigma: float = 1.5, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return X + rng.randn(*X.shape).astype(np.float32) * sigma


def _frequency_shift(X: np.ndarray, shift_hz: float = 3.0,
                     sfreq: float = 160.0) -> np.ndarray:
    """Add a sinusoidal carrier at shift_hz to every channel (phase modulation)."""
    t = np.linspace(0, X.shape[-1] / sfreq, X.shape[-1], dtype=np.float32)
    carrier = np.sin(2 * np.pi * shift_hz * t).astype(np.float32)
    return X + 0.5 * carrier[None, None, :]   # broadcast over batch × channels


def _channel_dropout(X: np.ndarray, drop_rate: float = 0.20,
                     seed: int = 0) -> np.ndarray:
    rng  = np.random.RandomState(seed)
    X_d  = X.copy()
    n_ch = X.shape[1]
    n_drop = max(1, int(n_ch * drop_rate))
    for i in range(len(X_d)):
        drop_ch = rng.choice(n_ch, n_drop, replace=False)
        X_d[i, drop_ch, :] = 0.0
    return X_d


CORRUPTIONS = {
    "noise_boost":      lambda X, sfreq: _noise_boost(X),
    "frequency_shift":  lambda X, sfreq: _frequency_shift(X, sfreq=sfreq),
    "channel_dropout":  lambda X, sfreq: _channel_dropout(X),
    "combined":         lambda X, sfreq: _channel_dropout(
                            _frequency_shift(_noise_boost(X), sfreq=sfreq)
                        ),
}


def run_domain_shift(model,
                     ds,
                     cfg:    TSTAConfig,
                     device: str,
                     subj:   int = 1) -> dict:
    """
    Args:
        model:  Trained TSTA model (trained on clean data)
        ds:     EEGDataset
        cfg:    TSTAConfig
        device: 'cpu' | 'cuda'
        subj:   Subject ID to use

    Returns:
        {
          'baseline_sdas':   float,
          'per_corruption':  {name: {'sdas': float, 'degradation_pct': float}},
          'mean_dt_sdas':    float,
          'claim_supported': bool,
        }
    """
    print("\n" + "=" * 60)
    print("  PHASE 3 — Domain Shift Simulation")
    print("=" * 60)

    mask = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]

    def _sdas(X_in):
        ds_ = TensorDataset(
            torch.tensor(X_in, dtype=torch.float32),
            torch.tensor(y_s,  dtype=torch.long),
        )
        dl = DataLoader(ds_, batch_size=cfg.BATCH_SIZE, shuffle=False)
        return compute_sdas(model, dl, device, cfg.N_CLASSES)["sdas"]

    baseline = _sdas(X_s)
    print(f"\n  Baseline SDAS (clean): {baseline:.4f}")
    print(f"\n  {'Corruption':<22}  {'SDAS':>8}  {'Δ%':>8}  Claim")
    print(f"  {'─'*56}")

    results = {}
    for name, corrupt_fn in CORRUPTIONS.items():
        X_c  = corrupt_fn(X_s, ds.sfreq)
        sdas = _sdas(X_c)
        deg  = (baseline - sdas) / (abs(baseline) + 1e-8) * 100
        ok   = sdas > cfg.TARGET_WITHIN_SDAS * 0.5    # half target still meaningful
        results[name] = {
            "sdas":            round(sdas, 4),
            "degradation_pct": round(deg, 2),
            "above_half_target": ok,
        }
        print(f"  {name:<22}  {sdas:>8.4f}  {deg:>+7.1f}%  "
              f"{'✓' if ok else '✗'}")

    dt_sdas  = float(np.mean([v["sdas"] for v in results.values()]))
    supported = dt_sdas > cfg.TARGET_WITHIN_SDAS * 0.5
    print(f"\n  Mean domain-transfer SDAS : {dt_sdas:.4f}")
    print(f"  Claim supported: {'✓ YES' if supported else '✗ NO'}"
          f"  (DT-SDAS > 0.5×target threshold)")

    return {
        "baseline_sdas":   round(baseline, 4),
        "per_corruption":  results,
        "mean_dt_sdas":    round(dt_sdas, 4),
        "claim_supported": supported,
    }
