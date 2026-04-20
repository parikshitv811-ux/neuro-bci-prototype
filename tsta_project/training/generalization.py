"""
Generalization & Zero-Shot Transfer Test
=========================================
Phase 8: Train on subjects [1,2,3], test on subject [4].
Evaluates how well subject-invariant directions transfer to unseen subjects.

Metrics:
  - SDAS on unseen subject
  - Direction similarity between train-set prototypes and test-set directions
  - Top-1 accuracy on unseen subject
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tsta_project.config           import TSTAConfig
from tsta_project.training.trainer import TSTATrainer
from tsta_project.training.metrics import compute_sdas


def run_generalization_test(ds,
                             cfg:          TSTAConfig,
                             device:       str,
                             train_subjs:  list = None,
                             test_subj:    int  = None,
                             epochs:       int  = 30) -> dict:
    """
    Train on `train_subjs`, evaluate on `test_subj` (zero-shot).

    Args:
        ds:           EEGDataset
        cfg:          TSTAConfig
        device:       compute device
        train_subjs:  list of subject IDs for training (None = all except last)
        test_subj:    subject ID for zero-shot test (None = last subject)
        epochs:       training epochs

    Returns:
        {
          'train_subjects':  list,
          'test_subject':    int,
          'train_sdas':      float,
          'zero_shot_sdas':  float,
          'zero_shot_top1':  float,
          'direction_sim':   float,  # mean cosine between train/test prototypes
        }
    """
    print("\n" + "=" * 60)
    print("  GENERALIZATION TEST — Zero-Shot Subject Transfer")
    print("=" * 60)

    unique = np.unique(ds.subjects).tolist()
    if train_subjs is None:
        train_subjs = unique[:-1]
    if test_subj is None:
        test_subj   = unique[-1]

    print(f"  Train subjects : {train_subjs}")
    print(f"  Test  subject  : {test_subj}")

    # ── Build train set ───────────────────────────────────────────────────────
    tr_mask  = np.isin(ds.subjects, train_subjs)
    X_tr, y_tr, s_tr = ds.X[tr_mask], ds.y[tr_mask], ds.subjects[tr_mask]

    trainer = TSTATrainer(cfg, device, n_subjects=len(train_subjs),
                          use_align=True, use_proto=True,
                          use_adv=True, use_smart=True)
    model, train_sdas = trainer.train(
        X_tr, y_tr, subjects=s_tr,
        epochs=epochs, tag="[Generalization]"
    )

    # ── Evaluate on training subjects ─────────────────────────────────────────
    tr_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                      torch.tensor(y_tr, dtype=torch.long)),
        batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False
    )
    train_m = compute_sdas(model, tr_dl, device, cfg.N_CLASSES)

    # ── Zero-shot eval on unseen subject ─────────────────────────────────────
    te_mask  = ds.subjects == test_subj
    X_te, y_te = ds.X[te_mask], ds.y[te_mask]

    if len(X_te) == 0:
        print(f"  ⚠ No data for test subject {test_subj}")
        return {"error": f"No data for subject {test_subj}"}

    te_dl = DataLoader(
        TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                      torch.tensor(y_te, dtype=torch.long)),
        batch_size=cfg.BATCH_SIZE, shuffle=False, drop_last=False
    )
    test_m = compute_sdas(model, te_dl, device, cfg.N_CLASSES)

    # ── Direction similarity: train vs test prototypes ────────────────────────
    model.eval()
    with torch.no_grad():
        ids  = torch.arange(cfg.N_CLASSES).to(device)
        txt  = F.normalize(model.encode_text(ids), dim=-1).cpu().numpy()

        # Compute mean direction per class on test set
        all_dirs, all_labels = [], []
        for x_b, y_b in te_dl:
            d, _, _ = model(x_b.to(device), y_b.to(device))
            all_dirs.append(F.normalize(d, dim=-1).cpu().numpy())
            all_labels.append(y_b.numpy())
        all_dirs   = np.concatenate(all_dirs)
        all_labels = np.concatenate(all_labels)

        # Compute per-class mean direction on test set
        dir_sims = []
        for c in range(cfg.N_CLASSES):
            mask = all_labels == c
            if mask.sum() > 0:
                mu  = all_dirs[mask].mean(axis=0)
                mu /= np.linalg.norm(mu) + 1e-8
                dir_sims.append(float(mu @ txt[c]))

    direction_sim = float(np.mean(dir_sims)) if dir_sims else 0.0

    result = {
        "train_subjects": train_subjs,
        "test_subject":   test_subj,
        "train_sdas":     round(train_m["sdas"],    4),
        "zero_shot_sdas": round(test_m["sdas"],     4),
        "zero_shot_top1": round(test_m["top1_acc"], 4),
        "direction_sim":  round(direction_sim,       4),
    }

    print(f"\n  Train  SDAS        : {result['train_sdas']:.4f}")
    print(f"  Test   SDAS (zero-shot): {result['zero_shot_sdas']:.4f}")
    print(f"  Test   Top-1       : {result['zero_shot_top1']*100:.1f}%")
    print(f"  Direction similarity: {result['direction_sim']:.4f}")
    ok = result["zero_shot_sdas"] > 0.10
    print(f"  {'✓ Transfer successful' if ok else '✗ Transfer below threshold'}")

    return result
