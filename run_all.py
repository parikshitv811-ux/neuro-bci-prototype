"""
TSTA — Master Runner
====================
Executes all 7 phases end-to-end and saves a full report.
"""
import sys, os, json, time, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/claude/tsta')

from phase1_data      import acquire_and_preprocess
from phase3_model     import TSTA, TSTAConfig
from phase4_5_train_eval import (run_within_subject, run_cross_subject,
                                  run_noise_robustness, run_ablation,
                                  compute_sdas)
from phase6_viz       import build_figure
from torch.utils.data import DataLoader, TensorDataset

def main():
    t_start = time.time()
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 65)
    print("  TSTA — FULL VALIDATION PIPELINE")
    print("  Temporal Semantic Trajectory Alignment")
    print("=" * 65)

    # ── PHASE 1+2: Data ───────────────────────────────────────────────────────
    print("\n[PHASE 1+2] Data acquisition & preprocessing")
    ds = acquire_and_preprocess(n_subjects=5, n_per_class=48)
    print(f"  Shape: {ds.X.shape}  |  source: {ds.source}")

    # ── CONFIG adapted to dataset ─────────────────────────────────────────────
    cfg = TSTAConfig()
    cfg.update_from_dataset(ds.X.shape[1], ds.sfreq, ds.X.shape[2])
    cfg.EPOCHS = 40
    print(f"\n  Config: {cfg.N_CHANNELS}ch  {cfg.SFREQ}Hz  "
          f"{cfg.N_SAMPLES}T  {cfg.N_PATCHES} patches  "
          f"D={cfg.D_MODEL}  {cfg.N_LAYERS}L")

    # ── PHASE 4A: Within-subject ──────────────────────────────────────────────
    print("\n[PHASE 4A] Within-subject training")
    ws_results, ws_models = run_within_subject(ds, cfg, device, epochs=cfg.EPOCHS)

    # Use subject 1's model as the primary model for viz/ablation
    primary_model = ws_models[1]

    # ── PHASE 4B: Cross-subject ───────────────────────────────────────────────
    print("\n[PHASE 4B] Cross-subject leave-one-out")
    cs_results = run_cross_subject(ds, cfg, device, epochs=30)

    # ── PHASE 5: Noise robustness ─────────────────────────────────────────────
    print("\n[PHASE 5] Noise robustness")
    noise_results = run_noise_robustness(primary_model, ds, cfg, device, subj=1)

    # ── PHASE 7: Ablation ─────────────────────────────────────────────────────
    print("\n[PHASE 7] Ablation study")
    ablation_results = run_ablation(ds, cfg, device, subj=1, epochs=25)

    # ── PHASE 6: Visualization ────────────────────────────────────────────────
    print("\n[PHASE 6] Generating visualizations")
    fig_path = build_figure(primary_model, ds, cfg, device,
                            ablation_results, noise_results, subj=1)

    # ── SUMMARY REPORT ────────────────────────────────────────────────────────
    elapsed  = time.time() - t_start
    ws_sdas  = [r['sdas']     for r in ws_results.values()]
    ws_top1  = [r['top1_acc'] for r in ws_results.values()]
    cs_sdas  = [r['sdas']     for r in cs_results.values()]
    cs_top1  = [r['top1_acc'] for r in cs_results.values()]

    report = {
        'hypothesis': 'EEG temporal dynamics encode semantic direction',
        'data_source': ds.source,
        'n_subjects': ds.n_subjects,
        'n_epochs_total': len(ds.y),

        'within_subject': {
            'mean_sdas':     round(np.mean(ws_sdas), 4),
            'std_sdas':      round(np.std(ws_sdas),  4),
            'mean_top1':     round(np.mean(ws_top1),  4),
            'per_subject':   {k: {'sdas':v['sdas'],'top1':v['top1_acc']}
                              for k,v in ws_results.items()},
            'target_0.4':    sum(1 for s in ws_sdas if s > 0.4),
            'publishable':   np.mean(ws_sdas) > 0.4,
        },
        'cross_subject': {
            'mean_sdas':     round(np.mean(cs_sdas), 4),
            'std_sdas':      round(np.std(cs_sdas),  4),
            'mean_top1':     round(np.mean(cs_top1),  4),
            'target_0.25':   sum(1 for s in cs_sdas if s > 0.25),
            'publishable':   np.mean(cs_sdas) > 0.25,
        },
        'noise_robustness': noise_results,
        'ablation': {k: {'sdas': v['sdas'], 'top1': v['top1_acc']}
                     for k, v in ablation_results.items()},
        'publishable_thresholds': {
            'within_sdas_>0.4':  np.mean(ws_sdas) > 0.4,
            'cross_sdas_>0.25':  np.mean(cs_sdas) > 0.25,
            'accuracy_>60pct':   np.mean(ws_top1) > 0.6,
            'all_met':           (np.mean(ws_sdas) > 0.4 and
                                  np.mean(cs_sdas) > 0.25 and
                                  np.mean(ws_top1) > 0.6),
        },
        'elapsed_s': round(elapsed, 1),
    }

    print("\n" + "=" * 65)
    print("  FINAL RESULTS")
    print("=" * 65)
    print(f"\n  Within-subject SDAS  : {report['within_subject']['mean_sdas']:.4f} "
          f"± {report['within_subject']['std_sdas']:.4f}  "
          f"{'✓ > 0.4' if report['within_subject']['publishable'] else '✗ < 0.4'}")
    print(f"  Cross-subject  SDAS  : {report['cross_subject']['mean_sdas']:.4f} "
          f"± {report['cross_subject']['std_sdas']:.4f}  "
          f"{'✓ > 0.25' if report['cross_subject']['publishable'] else '✗ < 0.25'}")
    print(f"  Within-subject Top-1 : {np.mean(ws_top1)*100:.1f}%  "
          f"{'✓ > 60%' if np.mean(ws_top1) > 0.6 else '✗ < 60%'}")
    print(f"\n  Publishable thresholds:")
    for k, v in report['publishable_thresholds'].items():
        print(f"    {'✓' if v else '✗'} {k.replace('_',' ')}")
    print(f"\n  Elapsed: {elapsed:.1f}s  |  Device: {device}")

    out_path = '/home/claude/tsta/tsta_report.json'
    with open(out_path, 'w') as f: json.dump(report, f, indent=2)
    print(f"\n  Report → {out_path}")
    print(f"  Figure → {fig_path}")
    return report

if __name__ == '__main__':
    main()
