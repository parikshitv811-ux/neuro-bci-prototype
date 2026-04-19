"""
Phase-Locked Temporal Attention (PLTA)
=======================================
Creates learnable soft gates centered at ERP latencies:
  - P2  gate  (200ms)
  - N2/P3 gate (300ms)
  - Late gate  (500ms)

Each gate is a Gaussian window over the patch time axis.
Scale parameters allow the model to downweight uninformative windows.

NOT MODIFIED from original TSTA architecture.
"""

import numpy as np
import torch
import torch.nn as nn
from tsta_project.config import TSTAConfig


class PLTA(nn.Module):
    """
    Phase-Locked Temporal Attention.

    Gates transformer tokens at physiologically meaningful latencies.
    All gate parameters are learnable.
    """

    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.cfg    = cfg
        n_gates     = len(cfg.PLTA_CENTERS_S)

        # Fixed patch center times (in seconds)
        patch_centers = [
            (i * cfg.PATCH_STEP + cfg.PATCH_LEN / 2) / cfg.SFREQ
            for i in range(cfg.N_PATCHES)
        ]
        self.register_buffer(
            "patch_centers",
            torch.tensor(patch_centers, dtype=torch.float32)
        )

        # Learnable gate centers (initialized at ERP latencies)
        self.gate_centers = nn.Parameter(
            torch.tensor(cfg.PLTA_CENTERS_S, dtype=torch.float32)
        )

        # Learnable gate widths (log-space for positivity)
        self.log_gate_width = nn.Parameter(
            torch.full((n_gates,), np.log(cfg.PLTA_WIDTH_S), dtype=torch.float32)
        )

        # Learnable gate scales — let model zero out uninformative windows
        self.gate_scales = nn.Parameter(torch.ones(n_gates))

        # Project G gated vectors back to D_MODEL
        self.gate_proj = nn.Linear(n_gates * cfg.D_MODEL, cfg.D_MODEL)
        self.norm      = nn.LayerNorm(cfg.D_MODEL)

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: (B, N_patches, D_MODEL)

        Returns:
            context: (B, D_MODEL)  — PLTA context vector
            gates:   (G, N_patches) — gate profiles for visualization
        """
        B, N, D = tokens.shape
        n_gates  = len(self.cfg.PLTA_CENTERS_S)

        # Gaussian gates: (G, N_patches)
        centers = self.gate_centers.unsqueeze(1)                             # (G, 1)
        widths  = torch.exp(self.log_gate_width).unsqueeze(1).clamp(0.02, 0.5)  # (G, 1)
        pc      = self.patch_centers.unsqueeze(0)                            # (1, N)
        gates   = torch.exp(-0.5 * ((pc - centers) / widths) ** 2)          # (G, N)
        gates   = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)
        scales  = torch.sigmoid(self.gate_scales).unsqueeze(1)
        gates   = gates * scales

        # Weighted sum of tokens for each gate
        gated = []
        for g in range(n_gates):
            w = gates[g].unsqueeze(0).unsqueeze(-1)   # (1, N, 1)
            gated.append((tokens * w).sum(dim=1))      # (B, D)
        gated = torch.cat(gated, dim=-1)               # (B, G*D)

        # Project back and add residual
        out = self.gate_proj(gated)
        out = self.norm(out + tokens.mean(dim=1))

        return out, gates.detach()
