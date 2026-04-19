"""
TSTA Trainer
=============
Handles within-subject and cross-subject leave-one-out training.
Saves the best model checkpoint (SDAS-based selection).
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from tsta_project.config    import TSTAConfig, MODELS_DIR
from tsta_project.model     import TSTA, infonce_loss
from tsta_project.training.metrics import compute_sdas


class TSTATrainer:
    """
    Full training loop with:
      - AdamW optimizer + CosineAnnealingLR scheduler
      - Best model saved by validation SDAS
      - Clean per-epoch logging
    """

    def __init__(self, cfg: TSTAConfig, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device

    def _make_loaders(self, X: np.ndarray, y: np.ndarray, val_ratio: float = 0.15):
        """Split data into train / validation dataloaders."""
        idx   = np.random.permutation(len(y))
        n_val = max(1, int(len(y) * val_ratio))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        def _mk(indices, drop_last: bool = True):
            ds = TensorDataset(
                torch.tensor(X[indices], dtype=torch.float32),
                torch.tensor(y[indices], dtype=torch.long),
            )
            return DataLoader(
                ds,
                batch_size=self.cfg.BATCH_SIZE,
                shuffle=True,
                drop_last=drop_last,
            )

        return _mk(tr_idx, drop_last=True), _mk(val_idx, drop_last=False)

    def train(self,
              X:      np.ndarray,
              y:      np.ndarray,
              epochs: int  = None,
              tag:    str  = "",
              save_path: str = None):
        """
        Train a TSTA model on data (X, y).

        Args:
            X:         (N, C, T) EEG data
            y:         (N,) class labels
            epochs:    Training epochs (defaults to cfg.EPOCHS)
            tag:       Display tag for logging
            save_path: If set, save best model here

        Returns:
            (model, best_sdas)
        """
        epochs   = epochs or self.cfg.EPOCHS
        tr_l, va_l = self._make_loaders(X, y)

        model = TSTA(self.cfg).to(self.device)
        opt   = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.LR, weight_decay=1e-4
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs
        )

        best_sdas  = -999.0
        best_state = None

        prefix = f"  {tag}" if tag else " "
        print(f"{prefix}  {'Ep':>4}  {'Loss':>8}  {'SDAS':>8}  {'Top-1':>7}")
        print(f"{prefix}  {'─'*38}")

        for ep in range(1, epochs + 1):
            model.train()
            total_loss = 0.0

            for x_b, y_b in tr_l:
                x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                opt.zero_grad()
                eeg_d, txt_e, _ = model(x_b, y_b)
                loss = infonce_loss(eeg_d, txt_e, model.log_temp)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()

            sched.step()

            if ep % 10 == 0 or ep == 1 or ep == epochs:
                m = compute_sdas(model, va_l, self.device, self.cfg.N_CLASSES)
                if m["sdas"] > best_sdas:
                    best_sdas  = m["sdas"]
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                flag = " ←" if m["sdas"] == best_sdas else ""
                print(
                    f"{prefix}  {ep:>4}  "
                    f"{total_loss/len(tr_l):>8.4f}  "
                    f"{m['sdas']:>8.4f}  "
                    f"{m['top1_acc']*100:>6.1f}%{flag}"
                )

        # Restore best checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save to disk if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  [Saved] {save_path}")

        return model, best_sdas
