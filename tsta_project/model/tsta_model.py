"""
Full TSTA Model
================
Assembles: EEGPatcher → TemporalTransformer → PLTA → TrajectoryHead
Trained with symmetric InfoNCE loss against learned text embeddings.

Architecture NOT MODIFIED from original.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsta_project.config import TSTAConfig
from tsta_project.model.patcher     import EEGPatcher
from tsta_project.model.transformer import TemporalTransformer
from tsta_project.model.plta        import PLTA
from tsta_project.model.trajectory  import TrajectoryHead


# ─── TEXT EMBEDDER ─────────────────────────────────────────────────────────────
class TextEmbedder(nn.Module):
    """
    Learned text embedding for each semantic intent class.
    Initialized orthogonally (maximally separated directions).
    """

    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.N_CLASSES, cfg.D_TEXT)
        nn.init.orthogonal_(self.embed.weight)
        self.proj = nn.Sequential(
            nn.Linear(cfg.D_TEXT, cfg.D_TEXT),
            nn.LayerNorm(cfg.D_TEXT),
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.embed(ids)), dim=-1)


# ─── FULL MODEL ────────────────────────────────────────────────────────────────
class TSTA(nn.Module):
    """
    Temporal Semantic Trajectory Alignment model.

    EEG path: x → Patcher → Transformer → PLTA → TrajectoryHead → direction
    Text path: class_id → TextEmbedder → text_emb

    Both outputs are L2-normalized unit vectors in a shared direction space.
    Trained with InfoNCE to align EEG directions to text embeddings.
    """

    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.cfg         = cfg
        self.patcher     = EEGPatcher(cfg)
        self.transformer = TemporalTransformer(cfg)
        self.plta        = PLTA(cfg)
        self.traj_head   = TrajectoryHead(cfg)
        self.text_embed  = TextEmbedder(cfg)
        self.log_temp    = nn.Parameter(torch.tensor(np.log(cfg.TEMPERATURE)))

    def encode_eeg(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, T)

        Returns:
            direction: (B, D_MODEL) — unit direction vector
            gates:     (G, N_patches) — PLTA gate profiles
        """
        tokens          = self.patcher(x)
        tokens          = self.transformer(tokens)
        plta_ctx, gates = self.plta(tokens)
        direction       = self.traj_head(tokens, plta_ctx)
        return direction, gates

    def encode_text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.text_embed(ids)

    def forward(self, x: torch.Tensor, ids: torch.Tensor):
        """
        Args:
            x:   (B, C, T) EEG epochs
            ids: (B,)      class labels

        Returns:
            eeg_dir:  (B, D_MODEL)
            text_emb: (B, D_TEXT)
            gates:    (G, N_patches)
        """
        eeg_dir, gates = self.encode_eeg(x)
        text_emb       = self.encode_text(ids)
        return eeg_dir, text_emb, gates

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── INFONCE LOSS ──────────────────────────────────────────────────────────────
def infonce_loss(eeg: torch.Tensor,
                 text: torch.Tensor,
                 log_temp: torch.Tensor) -> torch.Tensor:
    """
    Symmetric InfoNCE (CLIP-style) loss.

    Pulls EEG direction vectors toward their matching text embeddings.
    """
    temp   = torch.exp(log_temp).clamp(0.01, 1.0)
    sim    = torch.matmul(eeg, text.T) / temp
    labels = torch.arange(len(eeg), device=eeg.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
