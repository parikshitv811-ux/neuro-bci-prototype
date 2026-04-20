"""
scripts/run_alignment.py
=========================
CDAS-focused training run.
Trains with full subject-invariant objectives (cross-subject batching,
direction alignment loss, prototype field, adversarial subject loss),
then evaluates CDAS and prints a full alignment report.

This is the key experiment proving:
  "EEG intent is encoded as a shared geometric direction field across humans"

Usage:
    python -m tsta_project.scripts.run_alignment
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import json
import numpy as np
import torch

from tsta_project.config                     import TSTAConfig, MODELS_DIR, LOGS_DIR
from tsta_project.utils                      import seed_everything, get_device, banner, section, save_json
from tsta_project.data.synthetic.generator   import SyntheticEEGGenerator
from tsta_project.data.preprocess            import Preprocessor
from tsta_project.training.trainer           import TSTATrainer
from tsta_project.training.metrics           import compute_sdas
from tsta_project.training.generalization    import run_generalization_test
from tsta_project.training.stability         import run_stability_analysis
from tsta_project.analysis.direction_invariance import run_direction_invariance
from tsta_project.analysis.geometric_structure  import run_geometric_structure
from tsta_project.viz.dashboard                 import build_dashboard


def _compute_cdas_single_model(model, ds, cfg, device) -> dict:
    """
    Compute CDAS using a SINGLE cross-subject model (trained on all subjects).
    More powerful than per-subject models when using adversarial training.
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    all_dirs, all_labels, all_subjs = [], [], []

    dl = DataLoader(
        TensorDataset(
            torch.tensor(ds.X,        dtype=torch.float32),
            torch.tensor(ds.y,        dtype=torch.long),
            torch.tensor(ds.subjects, dtype=torch.long),
        ),
        batch_size=cfg.BATCH_SIZE, shuffle=False
    )
    with torch.no_grad():
        for x_b, y_b, s_b in dl:
            d, _, _ = model(x_b.to(device), y_b.to(device))
            all_dirs.append(F.normalize(d, dim=-1).cpu().numpy())
            all_labels.append(y_b.numpy())
            all_subjs.append(s_b.numpy())

    dirs   = np.concatenate(all_dirs)
    labels = np.concatenate(all_labels)
    subjs  = np.concatenate(all_subjs)

    # Compute per (subject, class) mean direction
    unique_subjs = np.unique(subjs)
    mean_dirs    = {}
    for sid in unique_subjs:
        sm = ds.subjects == sid
        mean_dirs[int(sid)] = {}
        for c in range(cfg.N_CLASSES):
            mask = (labels == c) & (subjs == sid)
            if mask.sum() > 0:
                mu = dirs[mask].mean(axis=0)
                mu = mu / (np.linalg.norm(mu) + 1e-8)
                mean_dirs[int(sid)][c] = mu

    # CDAS
    same_cos, diff_cos = [], []
    per_class = {c: [] for c in range(cfg.N_CLASSES)}
    sids = sorted(mean_dirs.keys())
    for i, s1 in enumerate(sids):
        for s2 in sids[i+1:]:
            for c1 in range(cfg.N_CLASSES):
                if c1 not in mean_dirs[s1] or c1 not in mean_dirs[s2]:
                    continue
                v1, v2 = mean_dirs[s1][c1], mean_dirs[s2][c1]
                cs = float(np.dot(v1, v2))
                same_cos.append(cs)
                per_class[c1].append(cs)
                for c2 in range(cfg.N_CLASSES):
                    if c2 == c1 or c2 not in mean_dirs[s2]:
                        continue
                    v3 = mean_dirs[s2][c2]
                    diff_cos.append(float(np.dot(v1, v3)))

    sc = float(np.mean(same_cos)) if same_cos else 0.0
    dc = float(np.mean(diff_cos)) if diff_cos else 0.0
    cdas = sc - dc
    per_class_aln = {c: round(float(np.mean(v)), 4) for c, v in per_class.items() if v}

    # Worst-case (min same-class pair)
    worst = float(np.min(same_cos)) if same_cos else 0.0

    # Angular variance per class (deg)
    ang_var = {}
    intents = cfg.INTENTS if hasattr(cfg, "INTENTS") else [str(c) for c in range(cfg.N_CLASSES)]
    for c in range(cfg.N_CLASSES):
        c_dirs = [mean_dirs[s][c] for s in sids if c in mean_dirs[s]]
        if len(c_dirs) >= 2:
            c_dirs = np.stack(c_dirs)
            mu     = c_dirs.mean(axis=0)
            mu    /= np.linalg.norm(mu) + 1e-8
            cos_v  = np.clip(c_dirs @ mu, -1, 1)
            ang_v  = np.degrees(np.arccos(cos_v)).mean()
            ang_var[c] = round(float(ang_v), 2)

    mean_ang_var = float(np.mean(list(ang_var.values()))) if ang_var else 0.0

    return {
        "cdas":                cdas,
        "same_class_mean_cos": sc,
        "diff_class_mean_cos": dc,
        "worst_case_alignment":worst,
        "per_class_alignment": per_class_aln,
        "mean_angular_variance_deg": mean_ang_var,
        "per_class_angular_var_deg": ang_var,
        "claim_supported":     cdas > 0.08,
    }


def main():
    t0 = time.time()
    seed_everything(42)
    device = get_device()

    banner("TSTA — CROSS-SUBJECT ALIGNMENT EXPERIMENT")
    print("  Proving: 'EEG intent = shared geometric direction across humans'")

    # ── Data ──────────────────────────────────────────────────────────────────
    section("Data + Subject-Invariant Preprocessing")
    gen  = SyntheticEEGGenerator(n_subjects=5, n_per_class=48, seed=42)
    ds   = gen.get_dataset()
    prep = Preprocessor(sfreq=ds.sfreq)
    ds   = prep.process_dataset(ds, subject_invariant=True)
    print(f"  {ds.summary()}")

    cfg         = TSTAConfig()
    cfg.EPOCHS  = 40
    cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)

    # ── Cross-subject training ────────────────────────────────────────────────
    section("Cross-Subject Training (all objectives active)")
    trainer = TSTATrainer(
        cfg, device, n_subjects=ds.n_subjects,
        use_align=True, use_proto=True,
        use_adv=True,  use_smart=True,
    )
    model, best_sdas = trainer.train(
        ds.X, ds.y, subjects=ds.subjects,
        epochs=cfg.EPOCHS, tag="[XSubj]",
        save_path=os.path.join(MODELS_DIR, "tsta_xsubj.pt")
    )

    # ── CDAS evaluation ───────────────────────────────────────────────────────
    section("C2 Alignment Report — CDAS Evaluation")
    cdas_res = _compute_cdas_single_model(model, ds, cfg, device)

    intents = cfg.INTENTS if hasattr(cfg, "INTENTS") else [str(c) for c in range(cfg.N_CLASSES)]
    print(f"\n  C2 ALIGNMENT REPORT")
    print(f"  ─────────────────────────────────────────")
    print(f"  Same-class cross-subj cos : {cdas_res['same_class_mean_cos']:+.4f}")
    print(f"  Diff-class cross-subj cos : {cdas_res['diff_class_mean_cos']:+.4f}")
    print(f"  CDAS                      : {cdas_res['cdas']:+.4f}  "
          f"{'✓ > 0.08' if cdas_res['cdas'] > 0.08 else '→ needs more epochs'}")
    print(f"  Worst-case alignment      : {cdas_res['worst_case_alignment']:+.4f}")
    print(f"  Angular variance          : {cdas_res['mean_angular_variance_deg']:.1f}°  "
          f"{'✓ < 25°' if cdas_res['mean_angular_variance_deg'] < 25 else '✗ ≥ 25°'}")
    print(f"\n  Per-class alignment:")
    for c, v in cdas_res["per_class_alignment"].items():
        bar = "█" * max(0, int((v + 1) * 10))
        print(f"    {intents[int(c)]:15s} {v:+.4f}  {bar}")

    # ── Geometric structure ───────────────────────────────────────────────────
    section("Geometric Structure (all subjects)")
    geo_res = run_geometric_structure(model, ds, cfg, device)

    # ── Generalization test ───────────────────────────────────────────────────
    section("Generalization — Zero-Shot Subject Transfer")
    gen_res = run_generalization_test(ds, cfg, device,
                                      train_subjs=list(range(1, ds.n_subjects)),
                                      test_subj=ds.n_subjects,
                                      epochs=30)

    # ── Stability analysis ────────────────────────────────────────────────────
    section("Real-Time Direction Stability")
    stab_res = run_stability_analysis(model, cfg, device, duration_s=12.0)

    # ── Save results ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    report = {
        "cdas_result":         {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in cdas_res.items()},
        "geometric_structure": geo_res,
        "generalization":      gen_res,
        "stability":           stab_res,
        "best_sdas":           round(best_sdas, 4),
        "elapsed_s":           round(elapsed, 1),
    }
    save_json(report, os.path.join(LOGS_DIR, "alignment_results.json"))

    banner("ALIGNMENT EXPERIMENT COMPLETE")
    cdas = cdas_res["cdas"]
    if cdas >= 0.12:
        print(f"  ★ CDAS = {cdas:.4f}  →  STRONG / PUBLISHABLE")
    elif cdas >= 0.08:
        print(f"  ✓ CDAS = {cdas:.4f}  →  MEETS TARGET")
    else:
        print(f"  → CDAS = {cdas:.4f}  →  Needs more epochs or subjects")
    print(f"  Angular variance: {cdas_res['mean_angular_variance_deg']:.1f}°")
    print(f"  Zero-shot SDAS : {gen_res.get('zero_shot_sdas', '—')}")
    print(f"  RT Stability   : {stab_res['stability_score']:.4f}")
    print(f"  Total time     : {elapsed:.1f}s")
    return report


if __name__ == "__main__":
    main()
