"""
scripts/run_advanced.py
========================
Advanced Research Analysis — 9-Phase Deep Validation

Runs all advanced analysis phases on a trained TSTA model
to prove or disprove the 5 deep claims about trajectory representation.

Requires: run_synthetic.py must have been run first (loads saved model).
          Falls back to training a fresh model if none found.

Usage:
    python -m tsta_project.scripts.run_advanced
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from tsta_project.config                    import TSTAConfig, MODELS_DIR, LOGS_DIR
from tsta_project.utils                     import seed_everything, get_device, banner, section, save_json
from tsta_project.data.synthetic.generator  import SyntheticEEGGenerator
from tsta_project.data.preprocess           import Preprocessor
from tsta_project.model                     import TSTA
from tsta_project.training.trainer          import TSTATrainer
from tsta_project.training.eval             import run_within_subject

# Analysis phases
from tsta_project.analysis.direction_invariance import run_direction_invariance
from tsta_project.analysis.amplitude_invariance import run_amplitude_invariance
from tsta_project.analysis.domain_shift         import run_domain_shift
from tsta_project.analysis.temporal_smoothness  import run_temporal_smoothness
from tsta_project.analysis.geometric_structure  import run_geometric_structure
from tsta_project.analysis.time_reversal        import run_time_reversal
from tsta_project.analysis.partial_signal       import run_partial_signal
from tsta_project.analysis.failure_modes        import run_failure_modes
from tsta_project.analysis.advanced_viz         import build_advanced_figures
from tsta_project.analysis.research_report      import print_research_report


def _load_or_train(cfg: TSTAConfig, ds, device: str) -> tuple:
    """
    Return (models_dict, primary_model).
    Loads saved models if available, otherwise trains fresh ones (fewer epochs for speed).
    """
    # Try to load saved models for subjects 1-5
    models = {}
    for sid in np.unique(ds.subjects):
        path = os.path.join(MODELS_DIR, f"tsta_subj{int(sid):02d}.pt")
        if os.path.exists(path):
            m = TSTA(cfg).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            m.eval()
            models[int(sid)] = m
            print(f"  [Loaded] Subject {int(sid):02d} model from {path}")

    if len(models) < len(np.unique(ds.subjects)):
        print(f"\n  Missing saved models — training fresh (fast: 20 epochs)...")
        _, models = run_within_subject(
            ds, cfg, device, epochs=20, save_dir=MODELS_DIR
        )
        models = {int(k): v for k, v in models.items()}

    primary = models[min(models.keys())]
    return models, primary


def main():
    t_start = time.time()
    seed_everything(42)
    device = get_device()

    banner("TSTA — ADVANCED RESEARCH ANALYSIS (9 Phases)")
    print(f"  Device : {device}")
    print(f"  Claims : 5 deep trajectory representation properties")
    print(f"  Phases : P1 Invariance  P2 Amplitude  P3 Domain Shift")
    print(f"           P4 Smoothness  P5 Geometry   P6 Causality")
    print(f"           P7 Early detect P8 Failures  P9 Visualisation")

    # ── Data ─────────────────────────────────────────────────────────────────
    section("Data — Synthetic EEG Dataset")
    gen = SyntheticEEGGenerator(n_subjects=5, n_per_class=48, seed=42)
    ds  = gen.get_dataset()
    print(f"  {ds.summary()}")

    prep = Preprocessor(sfreq=ds.sfreq)
    ds   = prep.process_dataset(ds)

    cfg = TSTAConfig()
    cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)
    print(f"  Config: {cfg}")

    # ── Load / train models ───────────────────────────────────────────────────
    section("Models — Load or Train Per-Subject Models")
    models, primary_model = _load_or_train(cfg, ds, device)
    intents = cfg.INTENTS

    all_results = {}

    # ── Phase 1: Direction invariance ─────────────────────────────────────────
    section("Phase 1 — Cross-Subject Direction Invariance")
    all_results["direction_invariance"] = run_direction_invariance(
        models, ds, cfg, device
    )

    # ── Phase 2: Amplitude invariance ─────────────────────────────────────────
    section("Phase 2 — Amplitude Invariance")
    all_results["amplitude_invariance"] = run_amplitude_invariance(
        primary_model, ds, cfg, device, subj=1
    )

    # ── Phase 3: Domain shift ─────────────────────────────────────────────────
    section("Phase 3 — Domain Shift Simulation")
    all_results["domain_shift"] = run_domain_shift(
        primary_model, ds, cfg, device, subj=1
    )

    # ── Phase 4: Temporal smoothness ──────────────────────────────────────────
    section("Phase 4 — Temporal Smoothness Analysis")
    all_results["temporal_smoothness"] = run_temporal_smoothness(
        primary_model, ds, cfg, device, subj=1
    )

    # ── Phase 5: Geometric structure ──────────────────────────────────────────
    section("Phase 5 — Geometric Structure of Direction Space")
    all_results["geometric_structure"] = run_geometric_structure(
        primary_model, ds, cfg, device, subj=1
    )

    # ── Phase 6: Time reversal ────────────────────────────────────────────────
    section("Phase 6 — Time Reversal (Causality Test)")
    all_results["time_reversal"] = run_time_reversal(
        primary_model, ds, cfg, device, subj=1
    )

    # ── Phase 7: Partial signal ───────────────────────────────────────────────
    section("Phase 7 — Partial Signal Robustness")
    all_results["partial_signal"] = run_partial_signal(
        primary_model, ds, cfg, device, subj=1
    )

    # ── Phase 8: Failure modes ────────────────────────────────────────────────
    section("Phase 8 — Failure Mode Analysis")
    all_results["failure_modes"] = run_failure_modes(
        primary_model, ds, cfg, device, subj=1
    )

    # ── Phase 9: Advanced visualisation ──────────────────────────────────────
    section("Phase 9 — Advanced Visualisation")
    saved_figs = build_advanced_figures(all_results, intents)

    # ── Final research report ─────────────────────────────────────────────────
    section("Final Research Report")
    report = print_research_report(all_results)

    # Save full results
    elapsed = time.time() - t_start
    all_results["elapsed_s"]  = round(elapsed, 1)
    all_results["n_figures"]  = len(saved_figs)
    all_results["figures"]    = saved_figs

    # Serialise: strip non-JSON-serialisable keys
    serialisable = {}
    for k, v in all_results.items():
        try:
            json.dumps(v)
            serialisable[k] = v
        except TypeError:
            pass

    save_json(serialisable, os.path.join(LOGS_DIR, "advanced_results.json"))
    print(f"\n  Total elapsed : {elapsed:.1f}s")
    print(f"  Figures saved : {len(saved_figs)}")
    print(f"  Results JSON  : {LOGS_DIR}/advanced_results.json")

    return all_results


if __name__ == "__main__":
    main()
