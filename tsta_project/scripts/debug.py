"""
scripts/debug.py
=================
Quick sanity checks:
  - Data generation (shape, values)
  - Model forward pass (no crash)
  - Single training step
  - SDAS computation
  - Config consistency

Usage:
    python -m tsta_project.scripts.debug
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsta_project.config                    import TSTAConfig
from tsta_project.utils                     import seed_everything, get_device, banner, section
from tsta_project.data.synthetic.generator  import SyntheticEEGGenerator
from tsta_project.data.preprocess           import Preprocessor
from tsta_project.model                     import TSTA, infonce_loss
from tsta_project.training.metrics          import compute_sdas


def run_debug():
    seed_everything(42)
    device = get_device()
    banner("TSTA — DEBUG / SANITY CHECK")
    print(f"  Device: {device}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    section("1. Data Generation")
    gen = SyntheticEEGGenerator(n_subjects=2, n_per_class=10, seed=42)
    ds  = gen.get_dataset()
    assert ds.X.shape == (100, 64, 320), f"Bad shape: {ds.X.shape}"
    assert ds.y.max() == 4
    print(f"  ✓ X shape : {ds.X.shape}")
    print(f"  ✓ y shape : {ds.y.shape}  unique: {np.unique(ds.y).tolist()}")
    print(f"  ✓ subjects: {np.unique(ds.subjects).tolist()}")

    # ── 2. Preprocessing ─────────────────────────────────────────────────────
    section("2. Preprocessing")
    prep = Preprocessor(sfreq=ds.sfreq)
    ds   = prep.process_dataset(ds, verbose=False)
    assert not np.isnan(ds.X).any(), "NaN after preprocessing!"
    print(f"  ✓ No NaN/Inf in preprocessed data")
    print(f"  ✓ Mean={ds.X.mean():.4f}  Std={ds.X.std():.4f}")

    # ── 3. Config ─────────────────────────────────────────────────────────────
    section("3. Config")
    cfg = TSTAConfig()
    cfg.update_from_dataset(ds.n_channels, ds.sfreq, ds.n_samples)
    assert cfg.N_PATCHES > 0, f"N_PATCHES={cfg.N_PATCHES}"
    print(f"  ✓ {cfg}")

    # ── 4. Model forward pass ─────────────────────────────────────────────────
    section("4. Model Forward Pass")
    model = TSTA(cfg).to(device)
    print(f"  ✓ Parameters: {model.n_params():,}")

    x   = torch.tensor(ds.X[:4], dtype=torch.float32).to(device)
    ids = torch.tensor(ds.y[:4], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        eeg_dir, text_emb, gates = model(x, ids)

    assert eeg_dir.shape  == (4, cfg.D_MODEL),  f"eeg_dir shape: {eeg_dir.shape}"
    assert text_emb.shape == (4, cfg.D_TEXT),    f"text_emb shape: {text_emb.shape}"
    assert gates.shape[0] == len(cfg.PLTA_CENTERS_S)
    print(f"  ✓ EEG direction : {eeg_dir.shape}  norm≈1: {eeg_dir.norm(dim=-1).mean():.4f}")
    print(f"  ✓ Text embedding: {text_emb.shape}")
    print(f"  ✓ PLTA gates    : {gates.shape}")

    # ── 5. Loss ───────────────────────────────────────────────────────────────
    section("5. InfoNCE Loss")
    loss = infonce_loss(eeg_dir, text_emb, model.log_temp)
    assert not torch.isnan(loss), "Loss is NaN!"
    print(f"  ✓ Loss = {loss.item():.4f}")

    # ── 6. Single optimizer step ──────────────────────────────────────────────
    section("6. Optimizer Step")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    opt.zero_grad()
    eeg_dir, text_emb, _ = model(x, ids)
    loss = infonce_loss(eeg_dir, text_emb, model.log_temp)
    loss.backward()
    opt.step()
    print(f"  ✓ Backward + optimizer step completed, loss={loss.item():.4f}")

    # ── 7. SDAS computation ───────────────────────────────────────────────────
    section("7. SDAS Computation")
    loader = DataLoader(
        TensorDataset(
            torch.tensor(ds.X, dtype=torch.float32),
            torch.tensor(ds.y, dtype=torch.long),
        ),
        batch_size=32,
        shuffle=False,
    )
    m = compute_sdas(model, loader, device, cfg.N_CLASSES)
    assert "sdas" in m
    print(f"  ✓ SDAS={m['sdas']:.4f}  Top-1={m['top1_acc']*100:.1f}%  "
          f"AntSep={m['antonym_sep']:.4f}")

    banner("ALL CHECKS PASSED ✓")
    return True


if __name__ == "__main__":
    run_debug()
