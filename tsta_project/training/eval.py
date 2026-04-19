"""
TSTA Evaluation — Within-Subject and Cross-Subject
====================================================
Phase 4A: within-subject training (one model per subject)
Phase 4B: cross-subject leave-one-out evaluation
Phase 5:  noise robustness
Ablation: TSTA full vs. No-PLTA vs. mean-pool (no trajectory)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tsta_project.config          import TSTAConfig
from tsta_project.model           import TSTA, infonce_loss
from tsta_project.training.trainer import TSTATrainer
from tsta_project.training.metrics import compute_sdas, noise_robustness


# ─── PHASE 4A: WITHIN-SUBJECT ──────────────────────────────────────────────────
def run_within_subject(ds, cfg: TSTAConfig, device: str, epochs: int = 40,
                       save_dir: str = None):
    """
    Train one model per subject on their own data.

    Returns:
        results: dict {subj_id: metrics_dict}
        models:  dict {subj_id: TSTA model}
    """
    print("\n" + "=" * 60)
    print("  PHASE 4A — Within-Subject Training")
    print("=" * 60)

    trainer = TSTATrainer(cfg, device)
    results, models = {}, {}

    for subj in np.unique(ds.subjects):
        mask = ds.subjects == subj
        X_s, y_s = ds.X[mask], ds.y[mask]

        save_path = (
            f"{save_dir}/tsta_subj{int(subj):02d}.pt"
            if save_dir else None
        )

        tag = f"[S{int(subj):02d}]"
        model, best_sdas = trainer.train(
            X_s, y_s, epochs=epochs, tag=tag, save_path=save_path
        )

        # Final full-set evaluation
        full_ds = TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long),
        )
        full_l = DataLoader(full_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        m = compute_sdas(model, full_l, device, cfg.N_CLASSES)

        results[int(subj)] = m
        models[int(subj)]  = model

        flag = "✓" if m["sdas"] > cfg.TARGET_WITHIN_SDAS else "✗"
        print(
            f"  {tag} SDAS={m['sdas']:.4f}  "
            f"Top-1={m['top1_acc']*100:.1f}%  "
            f"AntSep={m['antonym_sep']:.4f}  "
            f"{flag} (target > {cfg.TARGET_WITHIN_SDAS})"
        )

    sdas_vals = [r["sdas"] for r in results.values()]
    top1_vals = [r["top1_acc"] for r in results.values()]
    print(f"\n  Mean SDAS : {np.mean(sdas_vals):.4f} (σ={np.std(sdas_vals):.4f})")
    print(f"  Mean Top-1: {np.mean(top1_vals)*100:.1f}%")
    print(f"  Subjects > {cfg.TARGET_WITHIN_SDAS} SDAS: "
          f"{sum(1 for s in sdas_vals if s > cfg.TARGET_WITHIN_SDAS)}/{len(sdas_vals)}")

    return results, models


# ─── PHASE 4B: CROSS-SUBJECT LEAVE-ONE-OUT ─────────────────────────────────────
def run_cross_subject(ds, cfg: TSTAConfig, device: str, epochs: int = 30):
    """
    Leave-one-out cross-subject evaluation.
    Train on N-1 subjects, evaluate on the left-out subject.

    Returns:
        results: dict {test_subj_id: metrics_dict}
    """
    print("\n" + "=" * 60)
    print("  PHASE 4B — Cross-Subject (Leave-One-Out)")
    print("=" * 60)

    trainer  = TSTATrainer(cfg, device)
    subj_ids = np.unique(ds.subjects)
    results  = {}

    for test_subj in subj_ids:
        tr_mask = ds.subjects != test_subj
        te_mask = ds.subjects == test_subj
        X_tr, y_tr = ds.X[tr_mask], ds.y[tr_mask]
        X_te, y_te = ds.X[te_mask], ds.y[te_mask]

        tag = f"[LOO S{int(test_subj):02d}]"
        model, _ = trainer.train(X_tr, y_tr, epochs=epochs, tag=tag)

        te_ds = TensorDataset(
            torch.tensor(X_te, dtype=torch.float32),
            torch.tensor(y_te, dtype=torch.long),
        )
        te_l = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        m = compute_sdas(model, te_l, device, cfg.N_CLASSES)

        results[int(test_subj)] = m
        flag = "✓" if m["sdas"] > cfg.TARGET_CROSS_SDAS else "✗"
        print(
            f"  {tag} SDAS={m['sdas']:.4f}  "
            f"Top-1={m['top1_acc']*100:.1f}%  "
            f"{flag} (target > {cfg.TARGET_CROSS_SDAS})"
        )

    sdas_vals = [r["sdas"] for r in results.values()]
    top1_vals = [r["top1_acc"] for r in results.values()]
    print(f"\n  Mean cross-subject SDAS : {np.mean(sdas_vals):.4f}")
    print(f"  Mean cross-subject Top-1: {np.mean(top1_vals)*100:.1f}%")
    return results


# ─── PHASE 5: NOISE ROBUSTNESS ─────────────────────────────────────────────────
def run_noise_robustness(model, ds, cfg: TSTAConfig, device: str, subj: int = 1):
    """
    Measure SDAS as Gaussian noise sigma increases.

    Returns:
        dict: {sigma: sdas}
    """
    print("\n" + "=" * 60)
    print("  PHASE 5 — Noise Robustness")
    print("=" * 60)

    mask = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]
    full_l = DataLoader(
        TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long),
        ),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )

    nr = noise_robustness(model, full_l, device, cfg.NOISE_SIGMAS)

    print(f"\n  {'Noise σ':>9}  {'SDAS':>8}  Bar")
    print(f"  {'─'*40}")
    for sigma, sdas in nr.items():
        bar  = "█" * max(0, int(sdas * 20))
        note = " ← baseline" if sigma == 0.0 else ""
        print(f"  {sigma:>9.2f}  {sdas:>8.4f}  {bar}{note}")

    return nr


# ─── ABLATION ──────────────────────────────────────────────────────────────────
def run_ablation(ds, cfg: TSTAConfig, device: str, subj: int = 1, epochs: int = 25):
    """
    Ablation study comparing:
      1. Full TSTA
      2. No PLTA (mean pool replaces PLTA context)
      3. No trajectory (mean pool replaces displacement head)

    Returns:
        dict: {variant_name: metrics_dict}
    """
    print("\n" + "=" * 60)
    print("  ABLATION STUDY")
    print("=" * 60)

    mask = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]
    trainer = TSTATrainer(cfg, device)
    results = {}

    full_l = DataLoader(
        TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long),
        ),
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
    )

    # ① Full TSTA
    m, _ = trainer.train(X_s, y_s, epochs=epochs, tag="[Full TSTA   ]")
    results["TSTA (full)"] = compute_sdas(m, full_l, device, cfg.N_CLASSES)

    # ② No PLTA
    class TSTANoPLTA(TSTA):
        def encode_eeg(self, x):
            tokens    = self.patcher(x)
            tokens    = self.transformer(tokens)
            direction = self.traj_head(tokens, tokens.mean(dim=1))
            return direction, torch.zeros(1)

    m2 = TSTANoPLTA(cfg).to(device)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=cfg.LR)
    dl   = DataLoader(
        TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long),
        ),
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )
    for ep in range(epochs):
        m2.train()
        for x_b, y_b in dl:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt2.zero_grad()
            d, t, _ = m2(x_b, y_b)
            infonce_loss(d, t, m2.log_temp).backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
            opt2.step()
    results["No PLTA"] = compute_sdas(m2, full_l, device, cfg.N_CLASSES)

    # ③ No trajectory (mean pool)
    class TSTAMeanPool(TSTA):
        def encode_eeg(self, x):
            import torch.nn.functional as F_
            tokens    = self.patcher(x)
            tokens    = self.transformer(tokens)
            ctx, _    = self.plta(tokens)
            mean_tok  = tokens.mean(dim=1)
            direction = F_.normalize(
                self.traj_head.fuse(torch.cat([mean_tok, ctx], dim=-1)), dim=-1
            )
            return direction, torch.zeros(1)

    m3 = TSTAMeanPool(cfg).to(device)
    opt3 = torch.optim.AdamW(m3.parameters(), lr=cfg.LR)
    for ep in range(epochs):
        m3.train()
        for x_b, y_b in dl:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt3.zero_grad()
            d, t, _ = m3(x_b, y_b)
            infonce_loss(d, t, m3.log_temp).backward()
            torch.nn.utils.clip_grad_norm_(m3.parameters(), 1.0)
            opt3.step()
    results["No trajectory (mean pool)"] = compute_sdas(
        m3, full_l, device, cfg.N_CLASSES
    )

    # Summary
    print(f"\n  {'Variant':<30}  {'SDAS':>8}  {'Top-1':>7}  {'AntSep':>8}")
    print(f"  {'─'*60}")
    best_sdas = max(v["sdas"] for v in results.values())
    for name, r in results.items():
        flag = " ← best" if r["sdas"] == best_sdas else ""
        print(
            f"  {name:<30}  {r['sdas']:>8.4f}  "
            f"{r['top1_acc']*100:>6.1f}%  "
            f"{r['antonym_sep']:>8.4f}{flag}"
        )
    return results
