"""
Baseline Comparison
====================
Compares TSTA against three strong baselines:

  B1 — Random direction     : directions sampled from unit sphere
  B2 — Static embedding     : mean class embedding (no temporal info)
  B3 — CNN classifier       : simple 1D CNN trained with cross-entropy

Outputs per-subject SDAS for each baseline for statistical comparison.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tsta_project.config          import TSTAConfig
from tsta_project.training.metrics import compute_sdas


# ── Baseline 1: Random Directions ────────────────────────────────────────────

def _random_sdas(X: np.ndarray, y: np.ndarray,
                 cfg: TSTAConfig, device: str, n_trials: int = 5) -> float:
    """Mean SDAS when model outputs random unit vectors."""
    n_classes = cfg.N_CLASSES
    D         = cfg.D_MODEL

    sdas_list = []
    for _ in range(n_trials):
        dirs   = F.normalize(
            torch.randn(len(y), D), dim=-1
        )
        labels = torch.tensor(y, dtype=torch.long)
        ids    = torch.arange(n_classes)
        txt    = F.normalize(torch.randn(n_classes, D), dim=-1)

        sim    = torch.matmul(dirs, txt.T)
        cs     = sim[torch.arange(len(y)), labels].mean().item()
        mask   = torch.ones_like(sim, dtype=torch.bool)
        mask[torch.arange(len(y)), labels] = False
        ics    = sim[mask].mean().item()
        sdas_list.append(cs - ics)

    return float(np.mean(sdas_list))


# ── Baseline 2: Static Class Embedding ───────────────────────────────────────

def _static_sdas(X: np.ndarray, y: np.ndarray,
                 cfg: TSTAConfig, device: str) -> float:
    """
    Static embedding baseline: compute mean EEG amplitude per class (no temporal info).
    Map to direction via a fixed linear projection.
    """
    n_classes = cfg.N_CLASSES
    D         = cfg.D_MODEL

    # Use mean power per channel as static feature
    power = np.mean(X ** 2, axis=-1)          # (N, C)

    # Fit a simple linear map: class mean → direction
    class_means = []
    for c in range(n_classes):
        idx = y == c
        if idx.sum() > 0:
            class_means.append(power[idx].mean(axis=0))
        else:
            class_means.append(np.zeros(power.shape[-1]))
    class_means = np.stack(class_means)       # (C, CH)

    # Random projection to D dims (fixed seed)
    rng     = np.random.RandomState(0)
    proj    = rng.randn(power.shape[-1], D).astype(np.float32)
    dirs    = power @ proj                     # (N, D)
    dirs_t  = F.normalize(torch.tensor(dirs),  dim=-1)
    class_m = F.normalize(torch.tensor((class_means @ proj).astype(np.float32)), dim=-1)

    labels = torch.tensor(y, dtype=torch.long)
    sim    = torch.matmul(dirs_t, class_m.T)
    cs     = sim[torch.arange(len(y)), labels].mean().item()
    mask   = torch.ones_like(sim, dtype=torch.bool)
    mask[torch.arange(len(y)), labels] = False
    ics    = sim[mask].mean().item()
    return cs - ics


# ── Baseline 3: CNN Classifier ────────────────────────────────────────────────

class _SimpleCNN(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, n_classes: int, D: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=25, padding=12),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4, D),
            nn.LayerNorm(D),
        )

    def forward(self, x):
        return F.normalize(self.head(self.conv(x)), dim=-1)


def _cnn_sdas(X: np.ndarray, y: np.ndarray,
              cfg: TSTAConfig, device: str, epochs: int = 15) -> float:
    """Train a simple CNN with InfoNCE, report SDAS."""
    D      = cfg.D_MODEL
    model  = _SimpleCNN(X.shape[1], X.shape[2], cfg.N_CLASSES, D).to(device)
    text   = F.normalize(torch.randn(cfg.N_CLASSES, D), dim=-1).to(device)
    text.requires_grad_(True)
    opt    = torch.optim.Adam(list(model.parameters()) + [text], lr=3e-4)
    log_t  = torch.tensor(np.log(0.07)).to(device).requires_grad_(True)

    dl = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        ),
        batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=True
    )
    from tsta_project.model import infonce_loss
    model.train()
    for ep in range(epochs):
        for x_b, y_b in dl:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt.zero_grad()
            eeg = model(x_b)
            txt = F.normalize(text[y_b], dim=-1)
            infonce_loss(eeg, txt, log_t).backward()
            opt.step()

    model.eval()
    full_l = DataLoader(
        TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        ),
        batch_size=cfg.BATCH_SIZE, shuffle=False,
    )
    all_dirs, all_labels = [], []
    with torch.no_grad():
        text_mat = F.normalize(text, dim=-1)
        for x_b, y_b in full_l:
            all_dirs.append(model(x_b.to(device)).cpu())
            all_labels.append(y_b)
    all_dirs   = torch.cat(all_dirs)
    all_labels = torch.cat(all_labels)
    txt_cpu    = text_mat.cpu()
    sim        = torch.matmul(all_dirs, txt_cpu.T)
    cs         = sim[torch.arange(len(all_labels)), all_labels].mean().item()
    mask       = torch.ones_like(sim, dtype=torch.bool)
    mask[torch.arange(len(all_labels)), all_labels] = False
    ics        = sim[mask].mean().item()
    return cs - ics


# ── Master baseline runner ────────────────────────────────────────────────────

def run_baseline_comparison(ds,
                             cfg:    TSTAConfig,
                             device: str,
                             tsta_ws_results: dict = None) -> dict:
    """
    Run all three baselines per subject.

    Returns:
        {
          'random':         {subj: sdas, ...},
          'static':         {subj: sdas, ...},
          'cnn':            {subj: sdas, ...},
          'tsta':           {subj: sdas, ...},  (from ws_results if provided)
          'summary':        {method: mean_sdas},
        }
    """
    print("\n" + "=" * 60)
    print("  BASELINE COMPARISON")
    print("=" * 60)

    subj_ids = np.unique(ds.subjects)
    results  = {"random": {}, "static": {}, "cnn": {}}
    if tsta_ws_results:
        results["tsta"] = {int(k): v["sdas"] for k, v in tsta_ws_results.items()}

    print(f"\n  {'Subj':>5}  {'Random':>8}  {'Static':>8}  {'CNN':>8}"
          + (f"  {'TSTA':>8}" if tsta_ws_results else ""))
    print(f"  {'─'*52}")

    for sid in subj_ids:
        mask   = ds.subjects == sid
        X_s, y_s = ds.X[mask], ds.y[mask]

        r_sdas = _random_sdas(X_s, y_s, cfg, device)
        s_sdas = _static_sdas(X_s, y_s, cfg, device)
        c_sdas = _cnn_sdas(X_s, y_s, cfg, device, epochs=10)

        results["random"][int(sid)] = round(r_sdas, 4)
        results["static"][int(sid)] = round(s_sdas, 4)
        results["cnn"][int(sid)]    = round(c_sdas, 4)

        tsta_str = ""
        if tsta_ws_results:
            t_sdas   = results["tsta"].get(int(sid), float("nan"))
            tsta_str = f"  {t_sdas:>8.4f}"
        print(f"  {int(sid):>5}  {r_sdas:>8.4f}  {s_sdas:>8.4f}  {c_sdas:>8.4f}{tsta_str}")

    summary = {}
    for method in ("random", "static", "cnn"):
        vals = list(results[method].values())
        summary[method] = round(float(np.mean(vals)), 4)
    if tsta_ws_results:
        summary["tsta"] = round(float(np.mean(list(results["tsta"].values()))), 4)

    print(f"\n  {'Summary':>5}  ", end="")
    for m in ("random", "static", "cnn") + (("tsta",) if tsta_ws_results else ()):
        print(f"{summary.get(m, 0):>8.4f}  ", end="")
    print()

    results["summary"] = summary
    return results
