"""
scripts/demo_mode.py
=====================
One-shot demo: trains fast → runs all analysis → streams real-time → saves everything.
Perfect for showing the system to new users.

Usage:
    python -m tsta_project.scripts.demo_mode
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import torch
import numpy as np

from tsta_project.config                     import TSTAConfig, MODELS_DIR, LOGS_DIR
from tsta_project.utils                      import seed_everything, get_device, banner, section, save_json
from tsta_project.data.synthetic.generator   import SyntheticEEGGenerator
from tsta_project.data.preprocess            import Preprocessor
from tsta_project.training.trainer           import TSTATrainer
from tsta_project.training.eval              import run_within_subject, run_noise_robustness
from tsta_project.viz.dashboard              import build_dashboard
from tsta_project.analysis.direction_invariance import run_direction_invariance
from tsta_project.analysis.geometric_structure  import run_geometric_structure
from tsta_project.analysis.time_reversal        import run_time_reversal
from tsta_project.analysis.advanced_viz         import build_advanced_figures
from tsta_project.analysis.research_report      import print_research_report
from tsta_project.realtime.stream_simulator     import EEGStreamSimulator
from tsta_project.realtime.realtime_inference   import RealTimeInference
from tsta_project.data.synthetic.profiles       import CATEGORIES


def main():
    t0 = time.time()
    seed_everything(42)
    device = get_device()

    banner("TSTA — DEMO MODE  (fast · beautiful · complete)")
    print("  Training 3 subjects × 20 epochs → analysis → real-time sim → figures")

    # ── 1. Data ───────────────────────────────────────────────────────────────
    section("Step 1/6 — Synthetic Dataset")
    gen  = SyntheticEEGGenerator(n_subjects=3, n_per_class=32, seed=42)
    ds   = gen.get_dataset()
    prep = Preprocessor(sfreq=ds.sfreq)
    ds   = prep.process_dataset(ds)
    print(f"  {ds.summary()}")

    cfg = TSTAConfig()
    cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)
    cfg.EPOCHS = 20

    # ── 2. Train ──────────────────────────────────────────────────────────────
    section("Step 2/6 — Training (fast: 20 epochs)")
    ws_results, models = run_within_subject(ds, cfg, device, epochs=20,
                                             save_dir=MODELS_DIR)
    primary = models[min(models.keys())]

    sdas_vals = [r["sdas"] for r in ws_results.values()]
    top1_vals = [r["top1_acc"] for r in ws_results.values()]
    print(f"\n  SDAS: {np.mean(sdas_vals):.4f} ± {np.std(sdas_vals):.4f}  "
          f"Top-1: {np.mean(top1_vals)*100:.1f}%")

    # ── 3. Key analyses ───────────────────────────────────────────────────────
    section("Step 3/6 — Key Analysis Phases")
    adv = {}
    adv["direction_invariance"] = run_direction_invariance(models, ds, cfg, device)
    adv["geometric_structure"]  = run_geometric_structure(primary, ds, cfg, device)
    adv["time_reversal"]        = run_time_reversal(primary, ds, cfg, device)

    # ── 4. Noise robustness ───────────────────────────────────────────────────
    noise_res = run_noise_robustness(primary, ds, cfg, device, subj=1)

    # ── 5. Visualisations ─────────────────────────────────────────────────────
    section("Step 4/6 — Generating Figures")
    dash = build_dashboard(primary, ds, cfg, device,
                           ablation_results=None,
                           noise_results=noise_res, subj=1, tag="demo")
    figs = build_advanced_figures(adv, cfg.INTENTS)
    print(f"  Dashboard → {dash}")
    print(f"  Advanced  → {len(figs)} figures")

    # ── 6. Real-time simulation ───────────────────────────────────────────────
    section("Step 5/6 — Real-Time Simulation (8 seconds)")
    sim = EEGStreamSimulator(cfg, chunk_s=0.5, n_subjects=3, seed=42)
    rt  = RealTimeInference(primary, cfg, device, buffer_size=16)

    rt_log = []
    correct, total = 0, 0
    intent_cycle   = [0, 1, 2, 3, 4, 0, 1]
    for chunk, label, t_ms in sim.stream(duration_s=8.0, intent_seq=intent_cycle):
        r = rt.infer(chunk, true_label=label)
        rt_log.append({"t_ms": r["step"], "pred": r["pred_class"],
                       "true": label, "conf": r["confidence"]})
        if r["correct"] is not None:
            correct += int(r["correct"])
            total   += 1

    rt_acc = correct / max(total, 1)
    print(f"  Real-time accuracy: {rt_acc*100:.1f}%")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    section("Step 6/6 — Demo Summary")
    elapsed = time.time() - t0
    report = {
        "sdas_mean":     round(float(np.mean(sdas_vals)), 4),
        "sdas_std":      round(float(np.std(sdas_vals)),  4),
        "top1_mean":     round(float(np.mean(top1_vals)), 4),
        "rt_accuracy":   round(rt_acc, 4),
        "n_figures":     1 + len(figs),
        "cdas":          adv["direction_invariance"].get("cdas"),
        "geometry_ari":  adv["geometric_structure"].get("kmeans_ari"),
        "time_reversal_tre": adv["time_reversal"].get("tre"),
        "elapsed_s":     round(elapsed, 1),
        "mode":          "demo",
    }
    save_json(report, os.path.join(LOGS_DIR, "demo_results.json"))

    banner("DEMO COMPLETE")
    print(f"  SDAS   : {report['sdas_mean']:.4f} ± {report['sdas_std']:.4f}")
    print(f"  Top-1  : {report['top1_mean']*100:.1f}%")
    print(f"  RT Acc : {report['rt_accuracy']*100:.1f}%")
    print(f"  Figures: {report['n_figures']}")
    print(f"  Time   : {elapsed:.1f}s")
    return report


if __name__ == "__main__":
    main()
