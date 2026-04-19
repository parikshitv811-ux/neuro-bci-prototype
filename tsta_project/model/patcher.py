"""
EEG Patcher
===========
Splits an (C, T) EEG epoch into overlapping patches and projects
each patch to d_model via a shared linear layer.

NOT MODIFIED from original TSTA architecture.
"""

import torch
import torch.nn as nn
from tsta_project.config import TSTAConfig


class EEGPatcher(nn.Module):
    """
    Convert (B, C, T) tensor into (B, N_patches, D_MODEL) patch tokens.

    Steps:
      1. Slice T into N_patches overlapping windows of PATCH_LEN samples.
      2. Flatten each patch: (C × PATCH_LEN) → project → D_MODEL.
      3. Add learnable positional embeddings.
    """

    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.cfg      = cfg
        patch_dim     = cfg.N_CHANNELS * cfg.PATCH_LEN

        self.proj = nn.Sequential(
            nn.Linear(patch_dim, cfg.D_MODEL * 2),
            nn.GELU(),
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL),
        )
        self.pos_embed = nn.Parameter(
            torch.randn(cfg.N_PATCHES, cfg.D_MODEL) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)

        Returns:
            tokens: (B, N_patches, D_MODEL)
        """
        cfg = self.cfg
        patches = []
        for i in range(cfg.N_PATCHES):
            s = i * cfg.PATCH_STEP
            e = s + cfg.PATCH_LEN
            p = x[:, :, s:e].reshape(x.shape[0], -1)   # (B, C*patch_len)
            patches.append(p)
        patches = torch.stack(patches, dim=1)            # (B, N, C*patch_len)
        tokens  = self.proj(patches)                     # (B, N, D_MODEL)
        tokens  = tokens + self.pos_embed.unsqueeze(0)
        return tokens
