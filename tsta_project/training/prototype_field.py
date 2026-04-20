"""
Prototype Direction Field
==========================
Maintains a global direction prototype per semantic class that is updated
via EMA across the entire training run. Provides a stable target direction
that becomes more subject-invariant over time.

Architecture-safe: only operates on output direction vectors.
"""

import torch
import torch.nn.functional as F


class PrototypeField(torch.nn.Module):
    """
    Global direction prototype per class.

    Separate from DirectionAlignmentLoss (slower EMA, stable target).

    Args:
        n_classes:    Number of semantic classes
        d_model:      Direction dimension
        ema_alpha:    EMA rate (slow: 0.02)
        proto_weight: Loss weight
    """

    def __init__(self,
                 n_classes:    int,
                 d_model:      int,
                 ema_alpha:    float = 0.02,
                 proto_weight: float = 0.10):
        super().__init__()
        self.n_classes    = n_classes
        self.ema_alpha    = ema_alpha
        self.proto_weight = proto_weight

        # Stable global prototypes — orthogonalized, each row is a unit direction
        protos = torch.randn(n_classes, d_model)
        if d_model >= n_classes:
            q, _ = torch.linalg.qr(protos.T)       # q: (d_model, n_classes)
            protos = q.T[:n_classes].clone()         # (n_classes, d_model)
        self.register_buffer("v_class", F.normalize(protos, dim=-1))
        self._step = 0

    @torch.no_grad()
    def update(self, dirs: torch.Tensor, labels: torch.Tensor):
        """EMA update prototypes with current batch directions."""
        for c in range(self.n_classes):
            mask = labels == c
            if mask.sum() > 0:
                batch_mean = dirs[mask].mean(dim=0)
                batch_mean = F.normalize(batch_mean, dim=-1)
                alpha      = min(self.ema_alpha, self._step / (self._step + 50))
                updated    = (1 - alpha) * self.v_class[c] + alpha * batch_mean
                self.v_class[c] = F.normalize(updated, dim=-1)
        self._step += 1

    def forward(self, dirs: torch.Tensor, labels: torch.Tensor):
        """
        Compute prototype alignment loss.

        Args:
            dirs:   (B, D) normalized EEG direction vectors
            labels: (B,)   class labels

        Returns:
            loss scalar, metrics dict
        """
        dirs_n     = F.normalize(dirs, dim=-1)
        self.update(dirs_n.detach(), labels)

        protos     = self.v_class[labels]              # (B, D)
        cos_sim    = (dirs_n * protos).sum(dim=-1)     # (B,)
        l_proto    = (1.0 - cos_sim).mean()

        return self.proto_weight * l_proto, {"proto": l_proto.item()}

    @property
    def prototypes(self) -> torch.Tensor:
        """Return current class prototypes (n_classes, D)."""
        return self.v_class.clone()
