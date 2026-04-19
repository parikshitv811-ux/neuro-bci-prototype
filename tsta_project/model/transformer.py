"""
Temporal Transformer
=====================
Standard Pre-LN TransformerEncoder over patch tokens.

NOT MODIFIED from original TSTA architecture.
"""

import torch.nn as nn
from tsta_project.config import TSTAConfig


class TemporalTransformer(nn.Module):
    """
    Learns temporal relationships between patch tokens.

    Uses Pre-LayerNorm for training stability.
    """

    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.D_MODEL * 4,
            dropout=cfg.DROPOUT,
            batch_first=True,
            norm_first=True,     # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.N_LAYERS)
        self.norm    = nn.LayerNorm(cfg.D_MODEL)

    def forward(self, tokens):
        """
        Args:
            tokens: (B, N_patches, D_MODEL)

        Returns:
            tokens: (B, N_patches, D_MODEL)
        """
        return self.norm(self.encoder(tokens))
