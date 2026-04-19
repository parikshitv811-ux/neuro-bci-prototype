"""
scripts/run_synthetic.py
========================
Phase 0/1: Run the full TSTA pipeline on synthetic EEG data.

Steps:
  1. Generate synthetic dataset (5 subjects × 5 classes × 48 trials)
  2. Preprocess (bandpass + notch + baseline + z-score)
  3. Train TSTA (within-subject)
  4. Evaluate SDAS, Top-1, antonym separation
  5. Visualize dashboard
  6. Save model + results

Usage:
    python -m tsta_project.scripts.run_synthetic
    # or
    python tsta_project/scripts/run_synthetic.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsta_project.config                 import TSTAConfig, MODELS_DIR, LOGS_DIR
from tsta_project.utils                  import seed_everything, get_device, banner, section, save_json
from tsta_project.data.synthetic.generator import SyntheticEEGGenerator
from tsta_project.data.preprocess         import Preprocessor
from tsta_project.training.trainer        import TSTATrainer
from tsta_project.training.metrics        import compute_sdas, noise_robustness
from tsta_project.training.eval           import run_within_subject, run_noise_robustness
from tsta_project.viz.dashboard           import build_dashboard


def main():
    t_start = time.time()
    seed_everything(42)
    device  = get_device()

    banner("TSTA — SYNTHETIC PIPELINE")
    print(f"  Device : {device}")
    print(f"  Logs   : {LOGS_DIR}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    section("Step 1 — Generating Synthetic Dataset")
    gen = SyntheticEEGGenerator(n_subjects=5, n_per_class=48, seed=42)
    ds  = gen.get_dataset()
    print(f"  {ds.summary()}")

    section("Step 2 — Preprocessing")
    prep = Preprocessor(sfreq=ds.sfreq)
    ds   = prep.process_dataset(ds)
    print(f"  Mean={ds.X.mean():.4f}  Std={ds.X.std():.4f}")

    # ── 2. Config ─────────────────────────────────────────────────────────────
    cfg = TSTAConfig()
    cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)
    cfg.EPOCHS = 40
    print(f"\n  Config: {cfg}")

    # ── 3. Within-subject training ────────────────────────────────────────────
    ws_results, ws_models = run_within_subject(
        ds, cfg, device,
        epochs=cfg.EPOCHS,
        save_dir=MODELS_DIR,
    )

    # ── 4. Noise robustness (subject 1) ───────────────────────────────────────
    primary_model = ws_models[1]
    noise_res     = run_noise_robustness(primary_model, ds, cfg, device, subj=1)

    # ── 5. Visualization ──────────────────────────────────────────────────────
    section("Step 5 — Building Visualization Dashboard")
    fig_path = build_dashboard(
        primary_model, ds, cfg, device,
        ablation_results=None,
        noise_results=noise_res,
        subj=1,
        tag="synthetic",
    )

    # ── 6. Save primary model ─────────────────────────────────────────────────
    primary_path = os.path.join(MODELS_DIR, "tsta_synthetic.pt")
    torch.save(primary_model.state_dict(), primary_path)
    print(f"  [Model] Saved → {primary_path}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    elapsed   = time.time() - t_start
    sdas_vals = [r["sdas"]     for r in ws_results.values()]
    top1_vals = [r["top1_acc"] for r in ws_results.values()]

    banner("SYNTHETIC RESULTS")
    print(f"  SDAS (mean ± std) : {np.mean(sdas_vals):.4f} ± {np.std(sdas_vals):.4f}")
    print(f"  Top-1 accuracy    : {np.mean(top1_vals)*100:.1f}%")
    print(f"  Subjects > {cfg.TARGET_WITHIN_SDAS} SDAS : "
          f"{sum(1 for s in sdas_vals if s > cfg.TARGET_WITHIN_SDAS)}/{len(sdas_vals)}")

    check_ws  = np.mean(sdas_vals) > cfg.TARGET_WITHIN_SDAS
    check_top = np.mean(top1_vals) > cfg.TARGET_TOP1

    print(f"\n  {'✓' if check_ws  else '✗'} Within-subject SDAS > {cfg.TARGET_WITHIN_SDAS}")
    print(f"  {'✓' if check_top else '✗'} Top-1 accuracy   > {cfg.TARGET_TOP1*100:.0f}%")
    print(f"\n  Elapsed : {elapsed:.1f}s  |  Figure → {fig_path}")

    report = {
        "pipeline":        "synthetic",
        "n_subjects":      ds.n_subjects,
        "n_epochs":        ds.n_epochs,
        "within_subject":  {
            "mean_sdas":   round(float(np.mean(sdas_vals)), 4),
            "std_sdas":    round(float(np.std(sdas_vals)),  4),
            "mean_top1":   round(float(np.mean(top1_vals)), 4),
            "per_subject": {k: {"sdas": v["sdas"], "top1": v["top1_acc"]}
                            for k, v in ws_results.items()},
            "target_met":  check_ws,
        },
        "noise_robustness": noise_res,
        "elapsed_s":       round(elapsed, 1),
        "figure":          fig_path,
    }
    save_json(report, os.path.join(LOGS_DIR, "synthetic_results.json"))
    print(f"  Results → {LOGS_DIR}/synthetic_results.json")
    return report


if __name__ == "__main__":
    main()
