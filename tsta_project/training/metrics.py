"""
TSTA Evaluation Metrics
========================
  - SDAS (Semantic Direction Alignment Score)  — primary metric
  - Top-1 retrieval accuracy
  - Trajectory consistency (within-class cosine similarity)
  - Antonym separation score
  - Noise robustness (SDAS vs sigma)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_sdas(model, loader: DataLoader, device: str, n_classes: int = 5) -> dict:
    """
    Compute all evaluation metrics for a trained model.

    SDAS = mean(cos_sim_correct) − mean(cos_sim_incorrect)
    Range: [−1, 1].  Targets: within-subject > 0.4, cross-subject > 0.25.

    Args:
        model:     Trained TSTA model
        loader:    DataLoader with (x, y) batches
        device:    'cpu' or 'cuda'
        n_classes: Number of semantic classes

    Returns:
        dict with: sdas, top1_acc, correct_sim, incorrect_sim,
                   traj_cons (per class), antonym_sep
    """
    model.eval()
    all_eeg, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            eeg_dir, _, _ = model(x, y)
            all_eeg.append(eeg_dir.cpu())
            all_labels.append(y.cpu())

    all_eeg    = torch.cat(all_eeg)
    all_labels = torch.cat(all_labels)

    # Build full class text matrix
    ids = torch.arange(n_classes, device=device)
    with torch.no_grad():
        text_mat = model.encode_text(ids).cpu()   # (C, D)

    sim   = torch.matmul(all_eeg, text_mat.T)     # (N, C)
    preds = sim.argmax(dim=-1)
    top1  = (preds == all_labels).float().mean().item()

    correct_sim   = sim[torch.arange(len(all_labels)), all_labels].mean().item()
    mask          = torch.ones_like(sim, dtype=torch.bool)
    mask[torch.arange(len(all_labels)), all_labels] = False
    incorrect_sim = sim[mask].mean().item()
    sdas          = correct_sim - incorrect_sim

    # Trajectory consistency: within-class cosine similarity
    traj_cons = {}
    for cls in range(n_classes):
        idx = (all_labels == cls).nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue
        vecs    = F.normalize(all_eeg[idx], dim=-1)
        cos_mat = torch.matmul(vecs, vecs.T)
        off     = cos_mat[~torch.eye(len(vecs), dtype=torch.bool)]
        traj_cons[cls] = float(off.mean())

    # Antonym separation score
    antipodal = [(0, 1), (1, 0), (2, 3), (3, 2), (0, 4), (1, 4)]
    ant_sims  = []
    for a, b in antipodal:
        ant_sims.append(
            F.cosine_similarity(
                text_mat[a].unsqueeze(0),
                text_mat[b].unsqueeze(0),
            ).item()
        )
    antonym_sep = -float(np.mean(ant_sims))

    return {
        "sdas":          round(sdas, 4),
        "top1_acc":      round(top1, 4),
        "correct_sim":   round(correct_sim, 4),
        "incorrect_sim": round(incorrect_sim, 4),
        "traj_cons":     {k: round(v, 4) for k, v in traj_cons.items()},
        "antonym_sep":   round(antonym_sep, 4),
    }


def noise_robustness(model,
                     loader:       DataLoader,
                     device:       str,
                     noise_levels: list = None) -> dict:
    """
    Evaluate SDAS degradation as Gaussian noise is added to input.

    Args:
        model:        Trained TSTA model
        loader:       DataLoader
        device:       'cpu' | 'cuda'
        noise_levels: List of sigma values (default [0, 0.25, 0.5, 1.0, 2.0])

    Returns:
        dict: {sigma: sdas_value}
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.25, 0.5, 1.0, 2.0]

    model.eval()
    results = {}
    n_classes = model.cfg.N_CLASSES

    for sigma in noise_levels:
        all_eeg, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                xn = x + sigma * torch.randn_like(x)
                eeg_dir, _, _ = model(xn.to(device), y.to(device))
                all_eeg.append(eeg_dir.cpu())
                all_labels.append(y.cpu())

        all_eeg    = torch.cat(all_eeg)
        all_labels = torch.cat(all_labels)

        ids = torch.arange(n_classes, device=device)
        with torch.no_grad():
            text_mat = model.encode_text(ids).cpu()

        sim = torch.matmul(all_eeg, text_mat.T)
        cs  = sim[torch.arange(len(all_labels)), all_labels].mean().item()
        mask = torch.ones_like(sim, dtype=torch.bool)
        mask[torch.arange(len(all_labels)), all_labels] = False
        ics = sim[mask].mean().item()
        results[float(sigma)] = round(cs - ics, 4)

    return results
