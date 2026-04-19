"""
scripts/run_research.py
========================
Full research validation suite:
  1. Within-subject TSTA
  2. Baseline comparison (random, static, CNN)
  3. Statistical validation (t-tests, Wilcoxon, bootstrap CI)
  4. Advanced 9-phase analysis
  5. Publication figures

Usage:
    python -m tsta_project.scripts.run_research
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import torch
import numpy as np

from tsta_project.config                      import TSTAConfig, MODELS_DIR, LOGS_DIR
from tsta_project.utils                       import seed_everything, get_device, banner, section, save_json
from tsta_project.data.synthetic.generator   import SyntheticEEGGenerator
from tsta_project.data.preprocess            import Preprocessor
from tsta_project.training.eval              import run_within_subject
from tsta_project.research.statistical_validation import run_statistical_validation
from tsta_project.research.baselines         import run_baseline_comparison
from tsta_project.analysis                   import (
    run_direction_invariance, run_amplitude_invariance, run_domain_shift,
    run_temporal_smoothness, run_geometric_structure, run_time_reversal,
    run_partial_signal, run_failure_modes, print_research_report,
)
from tsta_project.analysis.advanced_viz      import build_advanced_figures


def main():
    t0 = time.time()
    seed_everything(42)
    device = get_device()
    banner("TSTA — RESEARCH VALIDATION SUITE")

    section("Data")
    gen  = SyntheticEEGGenerator(n_subjects=5, n_per_class=48, seed=42)
    ds   = gen.get_dataset()
    prep = Preprocessor(sfreq=ds.sfreq)
    ds   = prep.process_dataset(ds)

    cfg = TSTAConfig()
    cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)
    cfg.EPOCHS = 40

    section("Within-Subject Training")
    ws_results, models = run_within_subject(ds, cfg, device, epochs=cfg.EPOCHS,
                                             save_dir=MODELS_DIR)
    primary = models[min(models.keys())]

    section("Baseline Comparison")
    baseline_res = run_baseline_comparison(ds, cfg, device, ws_results)

    section("Statistical Validation")
    stat_res = run_statistical_validation(ws_results, {
        "Random":  list(baseline_res["random"].values()),
        "Static":  list(baseline_res["static"].values()),
        "CNN":     list(baseline_res["cnn"].values()),
    })

    section("Advanced 9-Phase Analysis")
    adv = {
        "direction_invariance": run_direction_invariance(models, ds, cfg, device),
        "amplitude_invariance": run_amplitude_invariance(primary, ds, cfg, device),
        "domain_shift":         run_domain_shift(primary, ds, cfg, device),
        "temporal_smoothness":  run_temporal_smoothness(primary, ds, cfg, device),
        "geometric_structure":  run_geometric_structure(primary, ds, cfg, device),
        "time_reversal":        run_time_reversal(primary, ds, cfg, device),
        "partial_signal":       run_partial_signal(primary, ds, cfg, device),
        "failure_modes":        run_failure_modes(primary, ds, cfg, device),
    }

    section("Advanced Visualisation")
    build_advanced_figures(adv, cfg.INTENTS)

    section("Research Report")
    report = print_research_report(adv)

    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s  |  Claims: {report['n_claims_supported']}/5")

    save_json({
        "within_subject": ws_results,
        "baselines": baseline_res,
        "statistics": stat_res,
        "advanced": adv,
        "elapsed_s": round(elapsed, 1),
    }, os.path.join(LOGS_DIR, "research_full_results.json"))

if __name__ == "__main__":
    main()
