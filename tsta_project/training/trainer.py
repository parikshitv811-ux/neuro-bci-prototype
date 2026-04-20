"""
TSTA Trainer — Cross-Subject Alignment Edition
================================================
Full training loop with subject-invariant auxiliary losses:
  - InfoNCE (core alignment)
  - DirectionAlignmentLoss (class prototype alignment + hard negatives)
  - PrototypeField (stable EMA global prototypes)
  - AdversarialSubjectLoss (gradient reversal, removes subject identity)
  - SubjectAwareBatchSampler (smart batching: same class, multiple subjects)

Architecture-safe: TSTA model itself is NOT modified.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tsta_project.config    import TSTAConfig, MODELS_DIR
from tsta_project.model     import TSTA, infonce_loss
from tsta_project.training.metrics import compute_sdas
from tsta_project.training.direction_alignment import DirectionAlignmentLoss
from tsta_project.training.prototype_field     import PrototypeField
from tsta_project.training.adversarial_subject import AdversarialSubjectLoss
from tsta_project.training.smart_batch         import SubjectAwareBatchSampler


class TSTATrainer:
    """
    Full training loop with cross-subject alignment objectives.

    Args:
        cfg:          TSTAConfig
        device:       'cpu' | 'cuda'
        n_subjects:   Number of subjects (for adversarial classifier)
        use_align:    Enable DirectionAlignmentLoss
        use_proto:    Enable PrototypeField
        use_adv:      Enable adversarial subject loss
        use_smart:    Enable subject-aware smart batching
    """

    def __init__(self,
                 cfg:         TSTAConfig,
                 device:      str  = "cpu",
                 n_subjects:  int  = 5,
                 use_align:   bool = True,
                 use_proto:   bool = True,
                 use_adv:     bool = True,
                 use_smart:   bool = True):
        self.cfg        = cfg
        self.device     = device
        self.n_subjects = n_subjects
        self.use_align  = use_align
        self.use_proto  = use_proto
        self.use_adv    = use_adv
        self.use_smart  = use_smart

    def _make_loaders(self,
                      X:        np.ndarray,
                      y:        np.ndarray,
                      subjects: np.ndarray = None,
                      val_ratio: float     = 0.15):
        """Build smart-batched train loader and standard validation loader."""
        idx    = np.random.permutation(len(y))
        n_val  = max(1, int(len(y) * val_ratio))
        val_i, tr_i = idx[:n_val], idx[n_val:]

        # Validation: standard loader (no smart batching needed)
        val_ds  = TensorDataset(
            torch.tensor(X[val_i],        dtype=torch.float32),
            torch.tensor(y[val_i],        dtype=torch.long),
        )
        val_l   = DataLoader(val_ds, batch_size=self.cfg.BATCH_SIZE,
                              shuffle=False, drop_last=False)

        # Training: smart batching if subjects provided and enabled
        X_tr = X[tr_i]; y_tr = y[tr_i]
        s_tr = subjects[tr_i] if subjects is not None else None

        if self.use_smart and s_tr is not None and len(np.unique(s_tr)) >= 2:
            tr_ds  = TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.long),
                torch.tensor(s_tr, dtype=torch.long),
            )
            sampler = SubjectAwareBatchSampler(
                y_tr, s_tr,
                batch_size=self.cfg.BATCH_SIZE,
                min_subjects_per_class=min(3, len(np.unique(s_tr))),
                n_per_class_per_subj=2,
                drop_last=True,
            )
            tr_l = DataLoader(tr_ds, batch_sampler=sampler)
        else:
            # Fallback: standard loader with subject column
            items = [torch.tensor(X_tr, dtype=torch.float32),
                     torch.tensor(y_tr, dtype=torch.long)]
            if s_tr is not None:
                items.append(torch.tensor(s_tr, dtype=torch.long))
            tr_ds = TensorDataset(*items)
            tr_l  = DataLoader(tr_ds, batch_size=self.cfg.BATCH_SIZE,
                                shuffle=True, drop_last=True)

        return tr_l, val_l

    def train(self,
              X:        np.ndarray,
              y:        np.ndarray,
              subjects: np.ndarray = None,
              epochs:   int        = None,
              tag:      str        = "",
              save_path: str       = None):
        """
        Train TSTA with full subject-invariant objective.

        Args:
            X:        (N, C, T) EEG data
            y:        (N,) class labels
            subjects: (N,) subject IDs (enables smart batching + adversarial)
            epochs:   Training epochs
            tag:      Display tag
            save_path: Save best model here

        Returns:
            (model, best_sdas)
        """
        epochs = epochs or self.cfg.EPOCHS

        # Determine actual subject count from data
        n_subjs = len(np.unique(subjects)) if subjects is not None else self.n_subjects
        n_subjs = max(n_subjs, 2)

        tr_l, va_l = self._make_loaders(X, y, subjects)

        model = TSTA(self.cfg).to(self.device)
        opt   = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.LR, weight_decay=1e-4
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        # Auxiliary loss modules (live on same device)
        align_loss = None
        proto_loss = None
        adv_loss   = None

        if self.use_align:
            align_loss = DirectionAlignmentLoss(
                self.cfg.N_CLASSES, self.cfg.D_MODEL,
                ema_alpha=0.08, align_weight=0.15, sep_weight=0.10,
                hard_weight=0.05,
            ).to(self.device)

        if self.use_proto:
            proto_loss = PrototypeField(
                self.cfg.N_CLASSES, self.cfg.D_MODEL,
                ema_alpha=0.02, proto_weight=0.10,
            ).to(self.device)

        if self.use_adv and subjects is not None:
            adv_loss = AdversarialSubjectLoss(
                self.cfg.D_MODEL, n_subjs,
                adv_weight=0.02, grl_alpha=1.0,
            ).to(self.device)
            # Optimizer must include adversarial classifier parameters
            opt = torch.optim.AdamW(
                list(model.parameters()) + list(adv_loss.parameters()),
                lr=self.cfg.LR, weight_decay=1e-4,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_sdas  = -999.0
        best_state = None
        prefix     = f"  {tag}" if tag else " "

        has_subj = subjects is not None
        print(f"{prefix}  {'Ep':>4}  {'Loss':>8}  {'SDAS':>8}  {'Top-1':>7}")
        print(f"{prefix}  {'─'*46}")

        for ep in range(1, epochs + 1):
            model.train()
            total_loss, n_batches = 0.0, 0

            for batch in tr_l:
                # Unpack (supports 2 or 3 tensor batches)
                if len(batch) == 3:
                    x_b, y_b, s_b = batch
                    x_b, y_b, s_b = (x_b.to(self.device),
                                      y_b.to(self.device),
                                      s_b.to(self.device))
                else:
                    x_b, y_b = batch
                    x_b, y_b = x_b.to(self.device), y_b.to(self.device)
                    s_b = None

                opt.zero_grad()
                eeg_d, txt_e, _ = model(x_b, y_b)

                # Core InfoNCE
                loss = infonce_loss(eeg_d, txt_e, model.log_temp)

                # Direction alignment + hard negatives
                if align_loss is not None:
                    l_al, _ = align_loss(eeg_d, y_b)
                    loss    = loss + l_al

                # Prototype field
                if proto_loss is not None:
                    l_pr, _ = proto_loss(eeg_d, y_b)
                    loss    = loss + l_pr

                # Adversarial subject
                if adv_loss is not None and s_b is not None:
                    # Remap subject IDs to 0-indexed
                    s_zero = s_b - s_b.min()
                    l_adv, _ = adv_loss(eeg_d, s_zero)
                    loss     = loss + l_adv

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                n_batches  += 1

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
                    f"{total_loss/max(n_batches,1):>8.4f}  "
                    f"{m['sdas']:>8.4f}  "
                    f"{m['top1_acc']*100:>6.1f}%{flag}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  [Saved] {save_path}")

        return model, best_sdas
