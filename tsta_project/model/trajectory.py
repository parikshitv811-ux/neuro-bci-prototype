"""
Trajectory Head
================
Computes a displacement-based direction vector from the sequence of
patch tokens. This is the core of TSTA:

  direction = Σ attn(disp_i) × disp_i  +  0.5 × (last_token − first_token)

Where disp_i = token[i+1] − token[i].

This captures WHERE the EEG trajectory went, not just its mean state.

NOT MODIFIED from original TSTA architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tsta_project.config import TSTAConfig


class TrajectoryHead(nn.Module):
    """
    Displacement-aware trajectory summary.

    Fuses:
      1. Attention-weighted sum of consecutive displacements.
      2. Global drift vector (last − first).
      3. PLTA context vector.
    """

    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.disp_attn = nn.Sequential(
            nn.Linear(cfg.D_MODEL, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Fuse displacement summary + PLTA context
        self.fuse = nn.Sequential(
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
            nn.GELU(),
            nn.LayerNorm(cfg.D_MODEL),
        )

    def forward(self, tokens: torch.Tensor, plta_context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens:       (B, N_patches, D_MODEL)
            plta_context: (B, D_MODEL)

        Returns:
            direction: (B, D_MODEL) — L2-normalized direction vector
        """
        disps    = tokens[:, 1:, :] - tokens[:, :-1, :]   # (B, N-1, D)
        attn_w   = torch.softmax(self.disp_attn(disps), dim=1)
        disp_sum = (attn_w * disps).sum(dim=1)             # (B, D)
        drift    = tokens[:, -1, :] - tokens[:, 0, :]     # (B, D)
        direction = disp_sum + 0.5 * drift

        direction = self.fuse(torch.cat([direction, plta_context], dim=-1))
        return F.normalize(direction, dim=-1)
