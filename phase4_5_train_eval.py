"""
TSTA Phase 4+5 — Training & Evaluation
=======================================
Phase 4: within-subject training + cross-subject leave-one-out
Phase 5: SDAS, top-1, cross-subject SDAS, trajectory consistency, noise robustness
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from phase3_model import TSTA, TSTAConfig, infonce_loss


# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_sdas(model, loader, device, n_classes=5):
    model.eval()
    all_eeg, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            eeg_dir, _, _ = model(x.to(device), y.to(device))
            all_eeg.append(eeg_dir.cpu()); all_labels.append(y.cpu())
    all_eeg    = torch.cat(all_eeg)
    all_labels = torch.cat(all_labels)

    # Full text matrix for all classes
    ids = torch.arange(n_classes).to(device)
    with torch.no_grad():
        text_mat = model.encode_text(ids).cpu()  # (C, D)

    sim     = torch.matmul(all_eeg, text_mat.T)  # (N, C)
    preds   = sim.argmax(dim=-1)
    top1    = (preds == all_labels).float().mean().item()

    correct_sim   = sim[torch.arange(len(all_labels)), all_labels].mean().item()
    mask          = torch.ones_like(sim, dtype=torch.bool)
    mask[torch.arange(len(all_labels)), all_labels] = False
    incorrect_sim = sim[mask].mean().item()
    sdas          = correct_sim - incorrect_sim

    # Trajectory consistency: within-class cosine similarity variance
    traj_cons = {}
    for cls in range(n_classes):
        idx = (all_labels == cls).nonzero(as_tuple=True)[0]
        if len(idx) < 2: continue
        vecs = F.normalize(all_eeg[idx], dim=-1)
        cos_mat = torch.matmul(vecs, vecs.T)
        off_diag = cos_mat[~torch.eye(len(vecs), dtype=torch.bool)]
        traj_cons[cls] = float(off_diag.mean())

    # Antonym separation
    antipodal = [(0,1),(1,0),(2,3),(3,2),(0,4),(1,4)]
    ant_sims  = []
    with torch.no_grad():
        for a, b in antipodal:
            va = text_mat[a]; vb = text_mat[b]
            ant_sims.append(F.cosine_similarity(va.unsqueeze(0),vb.unsqueeze(0)).item())
    antonym_sep = -float(np.mean(ant_sims))

    return {
        'sdas':         round(sdas, 4),
        'top1_acc':     round(top1, 4),
        'correct_sim':  round(correct_sim, 4),
        'incorrect_sim':round(incorrect_sim, 4),
        'traj_cons':    {k: round(v, 4) for k, v in traj_cons.items()},
        'antonym_sep':  round(antonym_sep, 4),
    }


def noise_robustness(model, loader, device, noise_levels=None):
    """Measure SDAS degradation as Gaussian noise is added."""
    if noise_levels is None:
        noise_levels = [0.0, 0.25, 0.5, 1.0, 2.0]
    results = {}
    model.eval()
    for sigma in noise_levels:
        noisy_ds = []
        for x, y in loader:
            xn = x + sigma * torch.randn_like(x)
            noisy_ds.append((xn, y))
        # Quick SDAS estimate on noisy data
        all_eeg, all_labels = [], []
        with torch.no_grad():
            for xn, y in noisy_ds:
                eeg_dir, _, _ = model(xn.to(device), y.to(device))
                all_eeg.append(eeg_dir.cpu()); all_labels.append(y.cpu())
        all_eeg    = torch.cat(all_eeg)
        all_labels = torch.cat(all_labels)
        ids = torch.arange(model.cfg.N_CLASSES).to(device)
        with torch.no_grad():
            text_mat = model.encode_text(ids).cpu()
        sim = torch.matmul(all_eeg, text_mat.T)
        cs  = sim[torch.arange(len(all_labels)), all_labels].mean().item()
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask[torch.arange(len(all_labels)), all_labels] = False
        ics = sim[mask].mean().item()
        results[sigma] = round(cs - ics, 4)
    return results


# ─── TRAINER ──────────────────────────────────────────────────────────────────
class TSTATrainer:
    def __init__(self, cfg: TSTAConfig, device='cpu'):
        self.cfg    = cfg
        self.device = device

    def _make_loaders(self, X, y, val_ratio=0.15):
        idx   = np.random.permutation(len(y))
        n_val = int(len(y) * val_ratio)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        def mk(i):
            ds = TensorDataset(torch.tensor(X[i]).float(),
                               torch.tensor(y[i]).long())
            return DataLoader(ds, batch_size=self.cfg.BATCH_SIZE,
                             shuffle=True, drop_last=True)
        return mk(tr_idx), mk(val_idx)

    def train(self, X, y, epochs=None, tag=''):
        epochs = epochs or self.cfg.EPOCHS
        tr_l, va_l = self._make_loaders(X, y)
        model = TSTA(self.cfg).to(self.device)
        opt   = torch.optim.AdamW(model.parameters(), lr=self.cfg.LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_sdas = -999; best_state = None

        print(f"\n  {tag}  {'Ep':>4}  {'Loss':>8}  {'SDAS':>8}  {'Top1':>7}")
        print(f"  {'─'*40}")

        for ep in range(1, epochs + 1):
            model.train(); total_loss = 0
            for x, y_b in tr_l:
                x, y_b = x.to(self.device), y_b.to(self.device)
                opt.zero_grad()
                eeg_d, txt_e, _ = model(x, y_b)
                loss = infonce_loss(eeg_d, txt_e, model.log_temp)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
            sched.step()

            if ep % 10 == 0 or ep == 1 or ep == epochs:
                m = compute_sdas(model, va_l, self.device, self.cfg.N_CLASSES)
                if m['sdas'] > best_sdas:
                    best_sdas  = m['sdas']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                flag = ' ←' if m['sdas'] == best_sdas else ''
                print(f"  {tag}  {ep:>4}  {total_loss/len(tr_l):>8.4f}  "
                      f"{m['sdas']:>8.4f}  {m['top1_acc']*100:>6.1f}%{flag}")

        model.load_state_dict(best_state)
        return model, best_sdas


# ─── PHASE 4A: WITHIN-SUBJECT ─────────────────────────────────────────────────
def run_within_subject(ds, cfg, device='cpu', epochs=50):
    print("\n" + "="*60)
    print("  PHASE 4A — Within-Subject Training")
    print("="*60)
    trainer = TSTATrainer(cfg, device)
    results = {}
    models  = {}

    for subj in np.unique(ds.subjects):
        mask = ds.subjects == subj
        X_s, y_s = ds.X[mask], ds.y[mask]
        tag = f'[S{subj}]'
        model, best_sdas = trainer.train(X_s, y_s, epochs=epochs, tag=tag)
        # Full eval
        full_ds = TensorDataset(torch.tensor(X_s).float(), torch.tensor(y_s).long())
        full_l  = DataLoader(full_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        m = compute_sdas(model, full_l, device, cfg.N_CLASSES)
        results[int(subj)] = m
        models[int(subj)]  = model
        print(f"  [S{subj}] Final SDAS={m['sdas']:.4f}  "
              f"Top1={m['top1_acc']*100:.1f}%  "
              f"AntSep={m['antonym_sep']:.4f}  "
              f"{'✓' if m['sdas'] > 0.4 else '✗'} (target > 0.4)")

    # Summary
    sdas_vals = [r['sdas'] for r in results.values()]
    top1_vals = [r['top1_acc'] for r in results.values()]
    print(f"\n  Within-subject mean SDAS : {np.mean(sdas_vals):.4f}  "
          f"(σ={np.std(sdas_vals):.4f})")
    print(f"  Within-subject mean Top1 : {np.mean(top1_vals)*100:.1f}%")
    print(f"  Subjects > 0.4 SDAS      : "
          f"{sum(1 for s in sdas_vals if s > 0.4)}/{len(sdas_vals)}")
    return results, models


# ─── PHASE 4B: CROSS-SUBJECT LEAVE-ONE-OUT ────────────────────────────────────
def run_cross_subject(ds, cfg, device='cpu', epochs=40):
    print("\n" + "="*60)
    print("  PHASE 4B — Cross-Subject (Leave-One-Out)")
    print("="*60)
    trainer   = TSTATrainer(cfg, device)
    subj_ids  = np.unique(ds.subjects)
    results   = {}

    for test_subj in subj_ids:
        tr_mask   = ds.subjects != test_subj
        te_mask   = ds.subjects == test_subj
        X_tr, y_tr = ds.X[tr_mask], ds.y[tr_mask]
        X_te, y_te = ds.X[te_mask], ds.y[te_mask]

        tag   = f'[LOO S{test_subj}]'
        model, _ = trainer.train(X_tr, y_tr, epochs=epochs, tag=tag)

        te_ds = TensorDataset(torch.tensor(X_te).float(), torch.tensor(y_te).long())
        te_l  = DataLoader(te_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
        m     = compute_sdas(model, te_l, device, cfg.N_CLASSES)
        results[int(test_subj)] = m
        print(f"  [LOO test S{test_subj}] SDAS={m['sdas']:.4f}  "
              f"Top1={m['top1_acc']*100:.1f}%  "
              f"{'✓' if m['sdas'] > 0.3 else '✗'} (target > 0.25–0.3)")

    sdas_vals = [r['sdas'] for r in results.values()]
    top1_vals = [r['top1_acc'] for r in results.values()]
    print(f"\n  Cross-subject mean SDAS : {np.mean(sdas_vals):.4f}")
    print(f"  Cross-subject mean Top1 : {np.mean(top1_vals)*100:.1f}%")
    return results


# ─── PHASE 5: NOISE ROBUSTNESS ────────────────────────────────────────────────
def run_noise_robustness(model, ds, cfg, device='cpu', subj=1):
    print("\n" + "="*60)
    print("  PHASE 5 — Noise Robustness")
    print("="*60)
    mask = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]
    full_l = DataLoader(
        TensorDataset(torch.tensor(X_s).float(), torch.tensor(y_s).long()),
        batch_size=cfg.BATCH_SIZE, shuffle=False
    )
    nr = noise_robustness(model, full_l, device,
                          noise_levels=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    print(f"\n  Noise σ   SDAS")
    print(f"  {'─'*22}")
    for sigma, sdas in nr.items():
        bar = '█' * max(0, int(sdas * 20))
        degraded = ' ← baseline' if sigma == 0.0 else ''
        print(f"  {sigma:>7.2f}   {sdas:>7.4f}  {bar}{degraded}")
    return nr


# ─── PHASE 7: ABLATION STUDY ─────────────────────────────────────────────────
def run_ablation(ds, cfg, device='cpu', subj=1, epochs=30):
    print("\n" + "="*60)
    print("  PHASE 7 — Ablation Study")
    print("="*60)
    mask = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]
    trainer  = TSTATrainer(cfg, device)
    results  = {}

    # ① Full TSTA
    m, _ = trainer.train(X_s, y_s, epochs=epochs, tag='[Full TSTA   ]')
    fl   = DataLoader(TensorDataset(torch.tensor(X_s).float(),
                                    torch.tensor(y_s).long()),
                      batch_size=cfg.BATCH_SIZE, shuffle=False)
    results['TSTA (full)'] = compute_sdas(m, fl, device, cfg.N_CLASSES)

    # ② No PLTA — standard transformer only
    class TSTANoPLTA(TSTA):
        def encode_eeg(self, x):
            tokens    = self.patcher(x)
            tokens    = self.transformer(tokens)
            direction = self.traj_head(tokens, tokens.mean(dim=1))
            return direction, torch.zeros(1)
    m2 = TSTANoPLTA(cfg).to(device)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=cfg.LR)
    for ep in range(epochs):
        m2.train()
        for x, yb in DataLoader(TensorDataset(torch.tensor(X_s).float(),
                                              torch.tensor(y_s).long()),
                                batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True):
            x, yb = x.to(device), yb.to(device)
            opt2.zero_grad()
            d, t, _ = m2(x, yb)
            infonce_loss(d, t, m2.log_temp).backward()
            torch.nn.utils.clip_grad_norm_(m2.parameters(), 1.0)
            opt2.step()
    results['No PLTA'] = compute_sdas(m2, fl, device, cfg.N_CLASSES)

    # ③ No trajectory — use mean pooling instead of displacement
    class TSTAMeanPool(TSTA):
        def encode_eeg(self, x):
            tokens   = self.patcher(x)
            tokens   = self.transformer(tokens)
            ctx, _   = self.plta(tokens)
            mean_tok = tokens.mean(dim=1)
            import torch.nn.functional as F_
            direction = F_.normalize(self.traj_head.fuse(
                torch.cat([mean_tok, ctx], dim=-1)), dim=-1)
            return direction, torch.zeros(1)
    m3 = TSTAMeanPool(cfg).to(device)
    opt3 = torch.optim.AdamW(m3.parameters(), lr=cfg.LR)
    for ep in range(epochs):
        m3.train()
        for x, yb in DataLoader(TensorDataset(torch.tensor(X_s).float(),
                                              torch.tensor(y_s).long()),
                                batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True):
            x, yb = x.to(device), yb.to(device)
            opt3.zero_grad()
            d, t, _ = m3(x, yb)
            infonce_loss(d, t, m3.log_temp).backward()
            torch.nn.utils.clip_grad_norm_(m3.parameters(), 1.0)
            opt3.step()
    results['No trajectory (mean pool)'] = compute_sdas(m3, fl, device, cfg.N_CLASSES)

    # Summary
    print(f"\n  {'Variant':<30}  {'SDAS':>8}  {'Top1':>7}  {'AntSep':>8}")
    print(f"  {'─'*60}")
    for name, r in results.items():
        flag = ' ← best' if r['sdas'] == max(v['sdas'] for v in results.values()) else ''
        print(f"  {name:<30}  {r['sdas']:>8.4f}  "
              f"{r['top1_acc']*100:>6.1f}%  {r['antonym_sep']:>8.4f}{flag}")
    return results
