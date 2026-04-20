"""
Direction Alignment Loss
=========================
Forces EEG direction vectors to align with a class-level mean direction
that is consistent across subjects.

Architecture-safe: only accesses output direction vectors, never touches
patcher/transformer/PLTA/trajectory head.

Losses:
  L_align = 1 - cosine(d, mu_class)         push toward class prototype
  L_sep   = max(0, cosine(d, mu_other) - margin)  push away from wrong class
  L_hard  = max(0, cosine(d, d_hard_neg) - margin) hard-negative mining
"""

import torch
import torch.nn.functional as F


class DirectionAlignmentLoss(torch.nn.Module):
    """
    Class-mean direction alignment with separation.

    Args:
        n_classes:    Number of semantic classes
        d_model:      Direction vector dimension
        ema_alpha:    EMA update rate for class prototypes
        align_weight: Weight for alignment loss
        sep_weight:   Weight for separation loss
        sep_margin:   Cosine margin for separation (lower = stricter)
        hard_margin:  Cosine margin for hard negatives
        hard_weight:  Weight for hard negative loss
    """

    def __init__(self,
                 n_classes:    int,
                 d_model:      int,
                 ema_alpha:    float = 0.05,
                 align_weight: float = 0.15,
                 sep_weight:   float = 0.10,
                 sep_margin:   float = 0.20,
                 hard_margin:  float = 0.30,
                 hard_weight:  float = 0.05):
        super().__init__()
        self.n_classes    = n_classes
        self.ema_alpha    = ema_alpha
        self.align_weight = align_weight
        self.sep_weight   = sep_weight
        self.sep_margin   = sep_margin
        self.hard_margin  = hard_margin
        self.hard_weight  = hard_weight

        # Learnable prototype directions (subject-invariant class directions)
        # Initialized orthogonally so classes start separated
        self.register_buffer(
            "class_protos",
            F.normalize(torch.randn(n_classes, d_model), dim=-1)
        )
        self._initialized = False

    @torch.no_grad()
    def _update_protos(self, dirs: torch.Tensor, labels: torch.Tensor):
        """EMA update of class prototype directions."""
        for c in range(self.n_classes):
            mask = labels == c
            if mask.sum() > 0:
                batch_mean = F.normalize(dirs[mask].mean(dim=0), dim=-1)
                if not self._initialized:
                    self.class_protos[c] = batch_mean
                else:
                    updated = ((1 - self.ema_alpha) * self.class_protos[c]
                               + self.ema_alpha * batch_mean)
                    self.class_protos[c] = F.normalize(updated, dim=-1)
        self._initialized = True

    def forward(self, dirs: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            dirs:   (B, D) normalized EEG direction vectors
            labels: (B,)   class labels

        Returns:
            total_loss, {align, sep, hard} dict
        """
        dirs_n = F.normalize(dirs, dim=-1)

        # Update EMA prototypes
        self._update_protos(dirs_n.detach(), labels)

        # ── Alignment loss ────────────────────────────────────────────────────
        protos_pos = self.class_protos[labels]          # (B, D)
        cos_pos    = (dirs_n * protos_pos).sum(dim=-1)  # (B,)
        l_align    = (1.0 - cos_pos).mean()

        # ── Separation loss ───────────────────────────────────────────────────
        # For each sample, average cosine with all other-class prototypes
        all_cos  = dirs_n @ self.class_protos.T          # (B, C)
        sep_loss = torch.zeros(1, device=dirs.device)
        for c in range(self.n_classes):
            mask     = labels != c
            if mask.sum() > 0:
                wrong    = torch.clamp(all_cos[mask, c] - self.sep_margin, min=0)
                sep_loss = sep_loss + wrong.mean()
        l_sep = sep_loss / self.n_classes

        # ── Hard negative mining ──────────────────────────────────────────────
        # Find most similar wrong-class direction in the batch
        B = dirs_n.shape[0]
        hard_loss = torch.zeros(1, device=dirs.device)
        if B > 1:
            sim_matrix = dirs_n @ dirs_n.T                # (B, B)
            for i in range(B):
                wrong_mask = labels != labels[i]
                if wrong_mask.sum() > 0:
                    hardest = sim_matrix[i][wrong_mask].max()
                    hard_loss = hard_loss + torch.clamp(
                        hardest - self.hard_margin, min=0
                    )
            hard_loss = hard_loss / B

        total = (self.align_weight * l_align
                 + self.sep_weight  * l_sep
                 + self.hard_weight * hard_loss)

        return total, {
            "align": l_align.item(),
            "sep":   l_sep.item(),
            "hard":  hard_loss.item(),
        }
