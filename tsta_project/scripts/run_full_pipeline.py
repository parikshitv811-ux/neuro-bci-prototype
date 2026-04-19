"""
scripts/run_full_pipeline.py
=============================
Master runner — all phases end-to-end.

Phases:
  0. Fix and validate synthetic pipeline
  1. Synthetic baseline (within-subject + noise)
  2. Real EEG pipeline (PhysioNet or synthetic fallback)
  3. Cross-subject generalization
  4. Ablation study
  5. Full visualization dashboard
  6. Final report

Usage:
    python -m tsta_project.scripts.run_full_pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import numpy as np
import torch

from tsta_project.config                    import TSTAConfig, MODELS_DIR, LOGS_DIR
from tsta_project.utils                     import seed_everything, get_device, banner, section, save_json, Timer
from tsta_project.data.synthetic.generator  import SyntheticEEGGenerator
from tsta_project.data.real.physionet_loader import acquire_dataset
from tsta_project.data.preprocess           import Preprocessor
from tsta_project.training.eval             import (
    run_within_subject, run_cross_subject,
    run_noise_robustness, run_ablation,
)
from tsta_project.viz.dashboard             import build_dashboard


def run_pipeline(ds_source: str = "auto",
                 n_subjects: int = 5,
                 n_per_class: int = 48,
                 ws_epochs: int = 40,
                 cs_epochs: int = 30,
                 abl_epochs: int = 25) -> dict:

    t_start = time.time()
    seed_everything(42)
    device  = get_device()

    banner("TSTA — FULL VALIDATION PIPELINE")
    print(f"  Device : {device}")
    print(f"  Source : {ds_source}")

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 0+1: Synthetic baseline (always run first)
    # ──────────────────────────────────────────────────────────────────────────
    banner("PHASE 0+1 — Synthetic Baseline")

    with Timer("Synthetic data generation"):
        gen  = SyntheticEEGGenerator(n_subjects=n_subjects,
                                     n_per_class=n_per_class, seed=42)
        ds_s = gen.get_dataset()
    print(f"  {ds_s.summary()}")

    with Timer("Preprocessing"):
        prep = Preprocessor(sfreq=ds_s.sfreq)
        ds_s = prep.process_dataset(ds_s)

    cfg_s = TSTAConfig()
    cfg_s.update_from_dataset(ds_s.n_channels, ds_s.sfreq, ds_s.n_samples)
    cfg_s.EPOCHS = ws_epochs

    with Timer("Within-subject training"):
        ws_syn, ws_models_syn = run_within_subject(
            ds_s, cfg_s, device, epochs=ws_epochs, save_dir=MODELS_DIR
        )

    primary_syn = ws_models_syn[1]

    with Timer("Noise robustness"):
        noise_syn = run_noise_robustness(primary_syn, ds_s, cfg_s, device, subj=1)

    # ──────────────────────────────────────────────────────────────────────────
    # PHASE 2+3: Real EEG (or synthetic fallback)
    # ──────────────────────────────────────────────────────────────────────────
    banner("PHASE 2+3 — Real EEG Pipeline")

    try:
        ds_r = acquire_dataset(source=ds_source,
                               n_subjects=n_subjects,
                               n_per_class=n_per_class)
    except Exception as e:
        print(f"  [FALLBACK] {e}")
        ds_r = SyntheticEEGGenerator(n_subjects=n_subjects,
                                     n_per_class=n_per_class,
                                     seed=1).get_dataset()

    print(f"  {ds_r.summary()}")
    prep_r = Preprocessor(sfreq=ds_r.sfreq)
    ds_r   = prep_r.process_dataset(ds_r)

    cfg_r = TSTAConfig()
    cfg_r.update_from_dataset(ds_r.n_channels, ds_r.sfreq, ds_r.n_samples)
    cfg_r.EPOCHS = ws_epochs

    with Timer("Within-subject (real)"):
        ws_real, ws_models_real = run_within_subject(
            ds_r, cfg_r, device, epochs=ws_epochs, save_dir=MODELS_DIR
        )

    with Timer("Cross-subject LOO"):
        cs_real = run_cross_subject(ds_r, cfg_r, device, epochs=cs_epochs)

    primary_real = ws_models_real[min(ws_models_real.keys())]
    noise_real   = run_noise_robustness(
        primary_real, ds_r, cfg_r, device,
        subj=int(min(np.unique(ds_r.subjects)))
    )

    # ──────────────────────────────────────────────────────────────────────────
    # ABLATION (on real/primary dataset, subject 1)
    # ──────────────────────────────────────────────────────────────────────────
    section("Ablation Study")
    ablation = run_ablation(
        ds_r, cfg_r, device,
        subj=int(min(np.unique(ds_r.subjects))),
        epochs=abl_epochs,
    )

    # ──────────────────────────────────────────────────────────────────────────
    # VISUALIZATION
    # ──────────────────────────────────────────────────────────────────────────
    section("Generating Dashboards")
    fig_syn = build_dashboard(
        primary_syn, ds_s, cfg_s, device,
        ablation_results=None,
        noise_results=noise_syn,
        subj=1, tag="synthetic",
    )
    fig_real = build_dashboard(
        primary_real, ds_r, cfg_r, device,
        ablation_results=ablation,
        noise_results=noise_real,
        subj=int(min(np.unique(ds_r.subjects))),
        tag=ds_r.source,
    )

    # Save primary models
    torch.save(primary_syn.state_dict(),
               os.path.join(MODELS_DIR, "tsta_synthetic.pt"))
    torch.save(primary_real.state_dict(),
               os.path.join(MODELS_DIR, "tsta_real.pt"))

    # ──────────────────────────────────────────────────────────────────────────
    # FINAL REPORT
    # ──────────────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    ws_sdas_syn  = [r["sdas"]     for r in ws_syn.values()]
    ws_top1_syn  = [r["top1_acc"] for r in ws_syn.values()]
    ws_sdas_real = [r["sdas"]     for r in ws_real.values()]
    ws_top1_real = [r["top1_acc"] for r in ws_real.values()]
    cs_sdas_real = [r["sdas"]     for r in cs_real.values()]

    banner("FINAL RESULTS")
    print("\n  SYNTHETIC:")
    print(f"    SDAS (within)  : {np.mean(ws_sdas_syn):.4f} ± {np.std(ws_sdas_syn):.4f}  "
          f"{'✓' if np.mean(ws_sdas_syn) > 0.4 else '✗'}")
    print(f"    Top-1          : {np.mean(ws_top1_syn)*100:.1f}%")

    print("\n  REAL EEG:")
    print(f"    SDAS (within)  : {np.mean(ws_sdas_real):.4f} ± {np.std(ws_sdas_real):.4f}  "
          f"{'✓' if np.mean(ws_sdas_real) > 0.4 else '✗'}")
    print(f"    SDAS (cross)   : {np.mean(cs_sdas_real):.4f}  "
          f"{'✓' if np.mean(cs_sdas_real) > 0.25 else '✗'}")
    print(f"    Top-1          : {np.mean(ws_top1_real)*100:.1f}%")

    all_met = (
        np.mean(ws_sdas_real) > 0.4
        and np.mean(cs_sdas_real) > 0.25
        and np.mean(ws_top1_real) > 0.6
    )
    print(f"\n  All publishable thresholds met: {'✓ YES' if all_met else '✗ NOT YET'}")
    print(f"  Elapsed: {elapsed:.1f}s  |  Device: {device}")

    report = {
        "hypothesis":   "EEG temporal dynamics encode semantic direction",
        "device":       device,
        "synthetic": {
            "mean_sdas":  round(float(np.mean(ws_sdas_syn)),  4),
            "mean_top1":  round(float(np.mean(ws_top1_syn)),  4),
            "noise":      noise_syn,
            "figure":     fig_syn,
        },
        "real_eeg": {
            "source":       ds_r.source,
            "within_sdas":  round(float(np.mean(ws_sdas_real)), 4),
            "cross_sdas":   round(float(np.mean(cs_sdas_real)), 4),
            "mean_top1":    round(float(np.mean(ws_top1_real)), 4),
            "noise":        noise_real,
            "figure":       fig_real,
        },
        "publishable_thresholds": {
            "within_sdas_>0.4":  bool(np.mean(ws_sdas_real) > 0.4),
            "cross_sdas_>0.25":  bool(np.mean(cs_sdas_real) > 0.25),
            "top1_>60pct":       bool(np.mean(ws_top1_real) > 0.6),
            "all_met":           bool(all_met),
        },
        "ablation":  {k: {"sdas": v["sdas"], "top1": v["top1_acc"]}
                      for k, v in ablation.items()},
        "elapsed_s": round(elapsed, 1),
    }
    report_path = os.path.join(LOGS_DIR, "full_pipeline_report.json")
    save_json(report, report_path)
    print(f"\n  Report → {report_path}")
    return report


if __name__ == "__main__":
    run_pipeline()
