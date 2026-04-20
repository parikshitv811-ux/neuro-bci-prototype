"""
Real-Time Direction Stability Analysis
========================================
Phase 9: Measures direction consistency during streaming inference.

Metrics:
  - Direction variance over time (per component)
  - Prediction stability: fraction of steps where class doesn't change
  - Stability Score = 1 / mean_angular_variance
  - Angular drift: max angle change between consecutive steps
"""

import numpy as np
import torch
import torch.nn.functional as F

from tsta_project.config                     import TSTAConfig
from tsta_project.realtime.stream_simulator  import EEGStreamSimulator
from tsta_project.realtime.realtime_inference import RealTimeInference


def run_stability_analysis(model,
                            cfg:        TSTAConfig,
                            device:     str,
                            duration_s: float = 12.0,
                            seed:       int   = 42) -> dict:
    """
    Stream EEG, collect direction sequences, measure temporal stability.

    Returns:
        {
          'mean_angular_variance_deg': float,
          'stability_score':           float,
          'prediction_stability':      float,
          'mean_angular_drift_deg':    float,
          'per_class_variance_deg':    {class_name: float},
        }
    """
    print("\n" + "=" * 60)
    print("  REAL-TIME STABILITY ANALYSIS")
    print("=" * 60)

    sim = EEGStreamSimulator(cfg, chunk_s=0.5, n_subjects=2, seed=seed)
    rt  = RealTimeInference(model, cfg, device, buffer_size=24, ema_alpha=0.3)

    directions = []  # (step, D)
    labels     = []
    preds      = []

    for chunk, label, t_ms in sim.stream(duration_s=duration_s):
        r = rt.infer(chunk, true_label=label)
        directions.append(np.array(r["smooth_dir"][:4]))   # first 4 dims for efficiency
        labels.append(label)
        preds.append(r["pred_class"])

    directions = np.array(directions)   # (T, 4)
    labels     = np.array(labels)
    preds      = np.array(preds)

    # Angular variance: mean angle change between consecutive steps
    if len(directions) >= 2:
        dots    = np.clip(
            np.sum(directions[:-1] * directions[1:], axis=-1) /
            (np.linalg.norm(directions[:-1], axis=-1) *
             np.linalg.norm(directions[1:],  axis=-1) + 1e-8),
            -1, 1
        )
        angles_deg     = np.degrees(np.arccos(dots))
        mean_drift_deg = float(angles_deg.mean())
        stability_score = 1.0 / (mean_drift_deg + 1e-3)
    else:
        mean_drift_deg  = 0.0
        stability_score = 0.0

    # Prediction stability: fraction where pred == prev pred
    if len(preds) >= 2:
        pred_stable = float((preds[1:] == preds[:-1]).mean())
    else:
        pred_stable = 1.0

    # Per-class angular variance
    unique_cls   = np.unique(labels)
    per_class_var = {}
    for c in unique_cls:
        mask = labels == c
        if mask.sum() >= 2:
            d_c = directions[mask]
            # Variance in direction = 1 - |mean_direction|
            mean_d = d_c.mean(axis=0)
            coherence = np.linalg.norm(mean_d) / (np.linalg.norm(d_c, axis=-1).mean() + 1e-8)
            angle_var = np.degrees(np.arccos(np.clip(coherence, -1, 1)))
            per_class_var[str(c)] = round(float(angle_var), 2)

    mean_angular_variance = float(np.mean(list(per_class_var.values()))) if per_class_var else 0.0

    result = {
        "mean_angular_variance_deg": round(mean_angular_variance, 2),
        "stability_score":           round(stability_score, 4),
        "prediction_stability":      round(pred_stable, 4),
        "mean_angular_drift_deg":    round(mean_drift_deg, 2),
        "per_class_variance_deg":    per_class_var,
        "n_steps":                   len(directions),
    }

    target_var = 25.0
    ok_var  = mean_angular_variance < target_var
    ok_stab = pred_stable > 0.6

    print(f"  Angular variance  : {mean_angular_variance:.1f}°  "
          f"{'✓ < 25°' if ok_var else '✗ ≥ 25°'}")
    print(f"  Stability score   : {stability_score:.4f}")
    print(f"  Prediction stable : {pred_stable*100:.1f}%  "
          f"{'✓' if ok_stab else '✗'}")
    print(f"  Mean angular drift: {mean_drift_deg:.1f}°/step")

    return result
