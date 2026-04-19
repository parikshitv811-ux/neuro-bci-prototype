"""
Phase 8 — Failure Mode Analysis
==================================
Identify where and why the trajectory model fails.

Analysis:
  • Collect all predictions and compare to ground truth.
  • For misclassified samples:
      - Which classes are most confused?
      - What is the trajectory smoothness vs correctly classified samples?
      - What is the magnitude of direction vectors?
      - Confusion matrix.
  • For each class: precision, recall, F1.
  • "Trajectory confidence" = max cosine similarity to any text embedding.

Outputs JSON + confusion matrix for the visualisation module.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tsta_project.config import TSTAConfig


def run_failure_modes(model,
                      ds,
                      cfg:    TSTAConfig,
                      device: str,
                      subj:   int = 1) -> dict:
    """
    Args:
        model:  Trained TSTA model
        ds:     EEGDataset
        cfg:    TSTAConfig
        device: 'cpu' | 'cuda'
        subj:   Subject to analyse

    Returns:
        {
          'accuracy':            float,
          'confusion_matrix':    list[list[int]],
          'per_class':           {cls: {precision, recall, f1, n_correct, n_total}},
          'mean_confidence_correct':   float,
          'mean_confidence_incorrect': float,
          'top_confusion_pairs':       list of (true, pred, count) sorted by count,
          'misclassified_indices':     list,
          'claim_supported':     bool,
        }
    """
    print("\n" + "=" * 60)
    print("  PHASE 8 — Failure Mode Analysis")
    print("=" * 60)

    mask   = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]

    model.eval()
    all_dirs, all_preds, all_labels, all_conf = [], [], [], []

    with torch.no_grad():
        ids      = torch.arange(cfg.N_CLASSES).to(device)
        text_mat = model.encode_text(ids)          # (C, D)

        ds_ = TensorDataset(
            torch.tensor(X_s, dtype=torch.float32),
            torch.tensor(y_s, dtype=torch.long),
        )
        dl  = DataLoader(ds_, batch_size=cfg.BATCH_SIZE, shuffle=False)

        for x_b, y_b in dl:
            d, _, _ = model(x_b.to(device), y_b.to(device))
            sim     = torch.matmul(d, text_mat.T)       # (B, C)
            preds   = sim.argmax(dim=-1)
            conf    = sim.max(dim=-1).values             # max cosine sim

            all_dirs.append(d.cpu())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_b.numpy())
            all_conf.append(conf.cpu().numpy())

    all_dirs   = F.normalize(torch.cat(all_dirs), dim=-1).numpy()
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_conf   = np.concatenate(all_conf)

    n_classes  = cfg.N_CLASSES
    intents    = cfg.INTENTS if hasattr(cfg, "INTENTS") else [str(c) for c in range(n_classes)]

    correct_mask   = (all_preds == all_labels)
    accuracy       = float(correct_mask.mean())
    misclassified  = np.where(~correct_mask)[0].tolist()

    # Confidence by outcome
    conf_correct   = float(all_conf[correct_mask].mean())  if correct_mask.sum()  > 0 else float('nan')
    conf_incorrect = float(all_conf[~correct_mask].mean()) if (~correct_mask).sum() > 0 else float('nan')

    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[int(t)][int(p)] += 1

    # Per-class precision / recall / F1
    per_class = {}
    print(f"\n  {'Class':<18}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'N':>5}")
    print(f"  {'─'*48}")
    for cls in range(n_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        per_class[cls] = {
            "precision": round(float(prec), 4),
            "recall":    round(float(rec),  4),
            "f1":        round(float(f1),   4),
            "n_correct": int(tp),
            "n_total":   int(cm[cls].sum()),
        }
        print(f"  {intents[cls]:<18}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  "
              f"{int(cm[cls].sum()):>5}")

    # Top confusion pairs
    pairs = []
    for t in range(n_classes):
        for p in range(n_classes):
            if t != p and cm[t, p] > 0:
                pairs.append((intents[t], intents[p], int(cm[t, p])))
    pairs.sort(key=lambda x: -x[2])

    print(f"\n  Top confusion pairs:")
    for t, p, cnt in pairs[:5]:
        print(f"    {t:<18} → {p:<18} ({cnt:>3} samples)")

    print(f"\n  Overall accuracy           : {accuracy*100:.1f}%")
    print(f"  Mean confidence (correct)  : {conf_correct:.4f}")
    print(f"  Mean confidence (incorrect): {conf_incorrect:.4f}"
          if not np.isnan(conf_incorrect) else "  (no incorrect predictions)")

    # Confidence gap > 0 means model is appropriately less confident when wrong
    conf_gap  = conf_correct - conf_incorrect if not np.isnan(conf_incorrect) else float('nan')
    supported = (accuracy > 0.6) and (np.isnan(conf_gap) or conf_gap > 0)
    print(f"  Claim supported: {'✓ YES' if supported else '✗ NO'}"
          f"  (accuracy > 60% and confidence gap > 0)")

    return {
        "accuracy":                    round(accuracy, 4),
        "confusion_matrix":            cm.tolist(),
        "per_class":                   {str(k): v for k, v in per_class.items()},
        "mean_confidence_correct":     round(conf_correct,   4),
        "mean_confidence_incorrect":   round(conf_incorrect, 4) if not np.isnan(conf_incorrect) else None,
        "top_confusion_pairs":         pairs[:10],
        "misclassified_indices":       misclassified[:50],
        "n_misclassified":             len(misclassified),
        "n_total":                     len(all_labels),
        "intents":                     intents,
        "claim_supported":             supported,
    }
