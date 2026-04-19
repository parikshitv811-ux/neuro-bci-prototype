"""
scripts/run_real.py
====================
Phase 2/3: Run the TSTA pipeline on real PhysioNet EEG.
Falls back to synthetic if PhysioNet is unavailable.

Steps:
  1. Load PhysioNet EEGMMIDB (auto-download via MNE) OR synthetic fallback
  2. Preprocess
  3. Train (within-subject + cross-subject)
  4. Evaluate
  5. Visualize

Usage:
    python -m tsta_project.scripts.run_real
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import numpy as np
import torch

from tsta_project.config                  import TSTAConfig, MODELS_DIR, LOGS_DIR
from tsta_project.utils                   import seed_everything, get_device, banner, section, save_json
from tsta_project.data.real.physionet_loader import acquire_dataset
from tsta_project.data.preprocess           import Preprocessor
from tsta_project.training.eval             import (
    run_within_subject, run_cross_subject, run_noise_robustness
)
from tsta_project.viz.dashboard             import build_dashboard


def main(n_subjects: int = 5, n_per_class: int = 48):
    t_start = time.time()
    seed_everything(42)
    device  = get_device()

    banner("TSTA — REAL EEG PIPELINE (PhysioNet / Fallback)")
    print(f"  Device : {device}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    section("Step 1 — Loading Data")
    try:
        ds = acquire_dataset(source="auto",
                             n_subjects=n_subjects,
                             n_per_class=n_per_class)
    except Exception as e:
        print(f"  [FALLBACK] {e}")
        from tsta_project.data.synthetic.generator import SyntheticEEGGenerator
        ds = SyntheticEEGGenerator(n_subjects=n_subjects,
                                   n_per_class=n_per_class).get_dataset()

    print(f"  {ds.summary()}")

    section("Step 2 — Preprocessing")
    prep = Preprocessor(sfreq=ds.sfreq)
    ds   = prep.process_dataset(ds)
    print(f"  Mean={ds.X.mean():.4f}  Std={ds.X.std():.4f}")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = TSTAConfig()
    cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)
    cfg.EPOCHS = 40
    print(f"\n  Config: {cfg}")

    # ── 3. Within-subject ────────────────────────────────────────────────────
    ws_results, ws_models = run_within_subject(
        ds, cfg, device,
        epochs=cfg.EPOCHS,
        save_dir=MODELS_DIR,
    )

    # ── 4. Cross-subject ─────────────────────────────────────────────────────
    cs_results = run_cross_subject(ds, cfg, device, epochs=30)

    # ── 5. Noise robustness ──────────────────────────────────────────────────
    primary_model = ws_models[min(ws_models.keys())]
    noise_res     = run_noise_robustness(
        primary_model, ds, cfg, device, subj=min(np.unique(ds.subjects))
    )

    # ── 6. Visualization ──────────────────────────────────────────────────────
    section("Step 6 — Building Visualization Dashboard")
    fig_path = build_dashboard(
        primary_model, ds, cfg, device,
        ablation_results=None,
        noise_results=noise_res,
        subj=min(np.unique(ds.subjects)),
        tag=ds.source,
    )

    # ── 7. Summary ────────────────────────────────────────────────────────────
    elapsed    = time.time() - t_start
    ws_sdas    = [r["sdas"]     for r in ws_results.values()]
    ws_top1    = [r["top1_acc"] for r in ws_results.values()]
    cs_sdas    = [r["sdas"]     for r in cs_results.values()]

    banner(f"REAL EEG RESULTS  [{ds.source.upper()}]")
    print(f"  Within-subject SDAS : {np.mean(ws_sdas):.4f} ± {np.std(ws_sdas):.4f}  "
          f"{'✓' if np.mean(ws_sdas) > cfg.TARGET_WITHIN_SDAS else '✗'} (target > {cfg.TARGET_WITHIN_SDAS})")
    print(f"  Cross-subject  SDAS : {np.mean(cs_sdas):.4f}  "
          f"{'✓' if np.mean(cs_sdas) > cfg.TARGET_CROSS_SDAS else '✗'} (target > {cfg.TARGET_CROSS_SDAS})")
    print(f"  Top-1 accuracy      : {np.mean(ws_top1)*100:.1f}%")
    print(f"  Elapsed : {elapsed:.1f}s  |  Figure → {fig_path}")

    report = {
        "pipeline":       ds.source,
        "n_subjects":     ds.n_subjects,
        "n_epochs":       ds.n_epochs,
        "within_subject": {
            "mean_sdas":   round(float(np.mean(ws_sdas)), 4),
            "std_sdas":    round(float(np.std(ws_sdas)),  4),
            "mean_top1":   round(float(np.mean(ws_top1)), 4),
            "target_met":  bool(np.mean(ws_sdas) > cfg.TARGET_WITHIN_SDAS),
        },
        "cross_subject": {
            "mean_sdas":   round(float(np.mean(cs_sdas)), 4),
            "target_met":  bool(np.mean(cs_sdas) > cfg.TARGET_CROSS_SDAS),
        },
        "noise_robustness": noise_res,
        "elapsed_s":       round(elapsed, 1),
        "figure":          fig_path,
    }
    save_json(report, os.path.join(LOGS_DIR, "real_results.json"))
    print(f"  Results → {LOGS_DIR}/real_results.json")
    return report


if __name__ == "__main__":
    main()
