"""
Phase 4 — Temporal Smoothness Analysis
=========================================
Claim:  Trajectory representations evolve smoothly over time for correctly
        predicted samples, and chaotically for incorrect predictions.

Method:
  • Extract intermediate patch token sequence from the transformer:
      tokens = Patcher(x) → Transformer(tokens)   shape (B, N_patches, D)
  • Compute displacement between consecutive tokens:
      disp_t = ||tokens[:, t+1, :] − tokens[:, t, :]||
  • Smoothness = mean displacement over all consecutive pairs.
  • Curvature  = mean ||disp_t+1 − disp_t|| (second derivative of trajectory).
  • Compare correct vs incorrect predictions.

No model modification required — we call model.patcher and model.transformer directly.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tsta_project.config import TSTAConfig


def _extract_patch_tokens(model, X: np.ndarray, y: np.ndarray,
                           cfg: TSTAConfig, device: str):
    """
    Returns:
        tokens:  (N, T_patches, D) numpy array of patch token sequences
        dirs:    (N, D)            final direction vectors
        preds:   (N,)              predicted class indices
        labels:  (N,)              ground-truth labels
    """
    model.eval()
    all_tokens, all_dirs, all_preds, all_labels = [], [], [], []

    with torch.no_grad():
        # Text embedding matrix for prediction
        ids      = torch.arange(cfg.N_CLASSES).to(device)
        text_mat = model.encode_text(ids)  # (C, D)

        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )
        dl = DataLoader(ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

        for x_b, y_b in dl:
            x_b = x_b.to(device)
            # Extract intermediate tokens (no modification to model)
            patch_tok  = model.patcher(x_b)          # (B, P, D)
            trans_tok  = model.transformer(patch_tok) # (B, P, D)
            dir_b, _, _ = model(x_b, y_b.to(device))

            # Predictions via cosine similarity
            sim   = torch.matmul(dir_b, text_mat.T)
            preds = sim.argmax(dim=-1)

            all_tokens.append(trans_tok.cpu().numpy())
            all_dirs.append(dir_b.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_b.numpy())

    tokens = np.concatenate(all_tokens, axis=0)  # (N, P, D)
    dirs   = np.concatenate(all_dirs,   axis=0)
    preds  = np.concatenate(all_preds,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return tokens, dirs, preds, labels


def _smoothness_and_curvature(tokens: np.ndarray) -> tuple[float, float]:
    """
    tokens: (N, P, D)
    Returns: (mean_smoothness, mean_curvature)
    """
    # displacement between consecutive patches: (N, P-1, D)
    disp    = tokens[:, 1:, :] - tokens[:, :-1, :]
    d_norms = np.linalg.norm(disp, axis=-1)         # (N, P-1)
    smoothness = float(d_norms.mean())

    # curvature: change in displacement direction (N, P-2, D)
    if disp.shape[1] >= 2:
        d_disp    = disp[:, 1:, :] - disp[:, :-1, :]
        c_norms   = np.linalg.norm(d_disp, axis=-1)
        curvature = float(c_norms.mean())
    else:
        curvature = 0.0

    return smoothness, curvature


def run_temporal_smoothness(model,
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
          'smoothness_correct':   float,
          'smoothness_incorrect': float,
          'curvature_correct':    float,
          'curvature_incorrect':  float,
          'per_class_smoothness': {cls: float},
          'smoothness_ratio':     float,  # incorrect / correct (>1 = correct smoother)
          'claim_supported':      bool,
        }
    """
    print("\n" + "=" * 60)
    print("  PHASE 4 — Temporal Smoothness Analysis")
    print("=" * 60)

    mask    = ds.subjects == subj
    X_s, y_s = ds.X[mask], ds.y[mask]
    tokens, dirs, preds, labels = _extract_patch_tokens(
        model, X_s, y_s, cfg, device
    )

    correct_mask   = (preds == labels)
    incorrect_mask = ~correct_mask

    n_correct   = int(correct_mask.sum())
    n_incorrect = int(incorrect_mask.sum())
    print(f"\n  Samples: {len(labels)}  correct={n_correct}  incorrect={n_incorrect}")

    sm_c, cu_c = _smoothness_and_curvature(tokens[correct_mask]) if n_correct > 0 else (0., 0.)
    sm_i, cu_i = _smoothness_and_curvature(tokens[incorrect_mask]) if n_incorrect > 0 else (float('nan'), float('nan'))

    print(f"\n  {'Group':<14}  {'Smoothness':>12}  {'Curvature':>12}")
    print(f"  {'─'*42}")
    print(f"  {'Correct':<14}  {sm_c:>12.6f}  {cu_c:>12.6f}")
    if n_incorrect > 0:
        print(f"  {'Incorrect':<14}  {sm_i:>12.6f}  {cu_i:>12.6f}")

    # Per-class smoothness
    per_class_sm = {}
    intents = cfg.INTENTS if hasattr(cfg, "INTENTS") else [str(c) for c in range(cfg.N_CLASSES)]
    print(f"\n  Per-class smoothness:")
    print(f"  {'Class':<18}  {'N':>5}  {'Smooth':>10}  {'Curvature':>10}")
    print(f"  {'─'*50}")
    for cls in range(cfg.N_CLASSES):
        idx = labels == cls
        if idx.sum() < 2:
            continue
        sm, cu = _smoothness_and_curvature(tokens[idx])
        per_class_sm[cls] = round(sm, 6)
        print(f"  {intents[cls]:<18}  {int(idx.sum()):>5}  {sm:>10.6f}  {cu:>10.6f}")

    ratio = (sm_i / sm_c) if (n_incorrect > 0 and sm_c > 0) else float('nan')
    supported = (n_incorrect == 0) or (not np.isnan(ratio) and ratio > 1.05)
    print(f"\n  Smoothness ratio (incorrect/correct) : {ratio:.4f}" if not np.isnan(ratio) else "")
    print(f"  Claim supported: {'✓ YES' if supported else '✗ NO'}"
          f"  (incorrect trajectories should be rougher)")

    return {
        "smoothness_correct":   round(sm_c, 6),
        "smoothness_incorrect": round(sm_i, 6) if not np.isnan(sm_i) else None,
        "curvature_correct":    round(cu_c, 6),
        "curvature_incorrect":  round(cu_i, 6) if not np.isnan(cu_i) else None,
        "per_class_smoothness": per_class_sm,
        "smoothness_ratio":     round(ratio, 4) if not np.isnan(ratio) else None,
        "claim_supported":      supported,
        "_tokens_sample":       tokens[:8].tolist(),  # for visualisation
        "_labels_sample":       labels[:8].tolist(),
        "_correct_mask":        correct_mask.tolist(),
    }
