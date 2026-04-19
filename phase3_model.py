"""
TSTA Phase 3 — Model with Phase-Locked Temporal Attention (PLTA)
=================================================================
Architecture:
  EEGPatcher → TemporalTransformer + PLTA gates → TrajectoryHead → InfoNCE

PLTA: gates attention at physiologically meaningful latencies
      (200ms P2, 300ms N2/P3, 500ms late component)
      Learnable: which window matters per-class is discovered during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── CONFIG ──────────────────────────────────────────────────────────────────
class TSTAConfig:
    # Incoming data (set by dataset)
    N_CHANNELS  = 64
    SFREQ       = 160
    N_SAMPLES   = 320          # 2s at 160Hz

    # Patching
    PATCH_LEN   = 40           # ~250ms at 160Hz
    PATCH_STEP  = 20           # ~125ms stride
    N_PATCHES   = (N_SAMPLES - PATCH_LEN) // PATCH_STEP + 1  # = 15

    # Model dims
    D_MODEL     = 128
    N_HEADS     = 4
    N_LAYERS    = 3
    D_TEXT      = 128
    DROPOUT     = 0.15

    # PLTA gate centers (in seconds) — P2, N2/P3, late component
    PLTA_CENTERS_S = [0.20, 0.30, 0.50]
    PLTA_WIDTH_S   = 0.08      # Gaussian gate half-width

    # Training
    BATCH_SIZE  = 32
    LR          = 3e-4
    EPOCHS      = 50
    TEMPERATURE = 0.07

    N_CLASSES   = 5
    INTENTS     = ['communication','navigation','action','selection','idle']

    def update_from_dataset(self, n_channels, sfreq, n_samples):
        self.N_CHANNELS = n_channels
        self.SFREQ      = sfreq
        self.N_SAMPLES  = n_samples
        self.PATCH_LEN  = max(16, int(0.25 * sfreq))
        self.PATCH_STEP = max(8,  int(0.125 * sfreq))
        self.N_PATCHES  = (n_samples - self.PATCH_LEN) // self.PATCH_STEP + 1


# ─── PATCHER ─────────────────────────────────────────────────────────────────
class EEGPatcher(nn.Module):
    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.cfg       = cfg
        patch_dim      = cfg.N_CHANNELS * cfg.PATCH_LEN
        self.proj      = nn.Sequential(
            nn.Linear(patch_dim, cfg.D_MODEL * 2),
            nn.GELU(),
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
            nn.LayerNorm(cfg.D_MODEL),
        )
        self.pos_embed = nn.Parameter(
            torch.randn(cfg.N_PATCHES, cfg.D_MODEL) * 0.02)

    def forward(self, x):
        # x: (B, C, T)
        cfg = self.cfg
        patches = []
        for i in range(cfg.N_PATCHES):
            s = i * cfg.PATCH_STEP
            e = s + cfg.PATCH_LEN
            p = x[:, :, s:e].reshape(x.shape[0], -1)  # (B, C*patch_len)
            patches.append(p)
        patches = torch.stack(patches, dim=1)           # (B, N_patches, C*patch_len)
        tokens  = self.proj(patches)                    # (B, N_patches, D_MODEL)
        tokens  = tokens + self.pos_embed.unsqueeze(0)
        return tokens


# ─── PHASE-LOCKED TEMPORAL ATTENTION (PLTA) ──────────────────────────────────
class PLTA(nn.Module):
    """
    Phase-Locked Temporal Attention.

    Creates learnable soft gates centered at ERP latencies.
    Each gate is a Gaussian window over patch time axis.
    Gates are modulated by a learned scale parameter — if a latency
    window is uninformative, the scale goes to zero.

    This forces the transformer to pay attention to specific time windows
    that align with known cognitive processing latencies.
    """
    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.cfg = cfg
        n_gates  = len(cfg.PLTA_CENTERS_S)

        # Compute patch center times (fixed, not learnable)
        patch_centers = []
        for i in range(cfg.N_PATCHES):
            s = i * cfg.PATCH_STEP
            center_s = (s + cfg.PATCH_LEN / 2) / cfg.SFREQ
            patch_centers.append(center_s)
        self.register_buffer('patch_centers',
                             torch.tensor(patch_centers, dtype=torch.float32))

        # Learnable gate centers (initialized at ERP latencies)
        init_centers = torch.tensor(cfg.PLTA_CENTERS_S, dtype=torch.float32)
        self.gate_centers = nn.Parameter(init_centers)

        # Learnable gate width (log-space for positivity)
        init_width = torch.full((n_gates,),
                               np.log(cfg.PLTA_WIDTH_S), dtype=torch.float32)
        self.log_gate_width = nn.Parameter(init_width)

        # Learnable gate scale per gate — lets model down-weight useless gates
        self.gate_scales = nn.Parameter(torch.ones(n_gates))

        # Project gated signal back to d_model
        self.gate_proj = nn.Linear(n_gates * cfg.D_MODEL, cfg.D_MODEL)
        self.norm      = nn.LayerNorm(cfg.D_MODEL)

    def forward(self, tokens):
        # tokens: (B, N_patches, D_MODEL)
        B, N, D = tokens.shape
        n_gates  = len(self.cfg.PLTA_CENTERS_S)

        # Compute Gaussian gates over patch axis: (n_gates, N_patches)
        centers = self.gate_centers.unsqueeze(1)        # (G, 1)
        widths  = torch.exp(self.log_gate_width).unsqueeze(1).clamp(0.02, 0.5)  # (G, 1)
        pc      = self.patch_centers.unsqueeze(0)       # (1, N)
        gates   = torch.exp(-0.5 * ((pc - centers) / widths) ** 2)  # (G, N)
        gates   = gates / (gates.sum(dim=1, keepdim=True) + 1e-8)   # normalize
        scales  = torch.sigmoid(self.gate_scales).unsqueeze(1)       # (G, 1)
        gates   = gates * scales                                      # (G, N)

        # Apply each gate to token sequence: weighted sum over time
        gated = []
        for g in range(n_gates):
            w = gates[g].unsqueeze(0).unsqueeze(-1)   # (1, N, 1)
            gated.append((tokens * w).sum(dim=1))      # (B, D)
        gated = torch.cat(gated, dim=-1)               # (B, G*D)

        # Project back and add residual via mean-pool of tokens
        out = self.gate_proj(gated)                    # (B, D)
        out = self.norm(out + tokens.mean(dim=1))      # residual

        return out, gates.detach()  # also return gates for visualization


# ─── TEMPORAL TRANSFORMER ────────────────────────────────────────────────────
class TemporalTransformer(nn.Module):
    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D_MODEL, nhead=cfg.N_HEADS,
            dim_feedforward=cfg.D_MODEL * 4,
            dropout=cfg.DROPOUT, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.N_LAYERS)
        self.norm    = nn.LayerNorm(cfg.D_MODEL)

    def forward(self, tokens):
        return self.norm(self.encoder(tokens))


# ─── TRAJECTORY HEAD ─────────────────────────────────────────────────────────
class TrajectoryHead(nn.Module):
    """
    Displacement-based trajectory summary.
    direction = Σ attention(disp_i) * disp_i  +  0.5 * (last - first)
    """
    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.disp_attn = nn.Sequential(
            nn.Linear(cfg.D_MODEL, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # PLTA injects a context vector; fuse it with displacement summary
        self.fuse = nn.Sequential(
            nn.Linear(cfg.D_MODEL * 2, cfg.D_MODEL),
            nn.GELU(),
            nn.LayerNorm(cfg.D_MODEL),
        )

    def forward(self, tokens, plta_context):
        # tokens: (B, N, D),  plta_context: (B, D)
        disps     = tokens[:, 1:, :] - tokens[:, :-1, :]  # (B, N-1, D)
        attn_w    = torch.softmax(self.disp_attn(disps), dim=1)
        disp_sum  = (attn_w * disps).sum(dim=1)            # (B, D)
        drift     = tokens[:, -1, :] - tokens[:, 0, :]    # (B, D)
        direction = disp_sum + 0.5 * drift                 # (B, D)
        # Fuse with PLTA temporal context
        direction = self.fuse(torch.cat([direction, plta_context], dim=-1))
        return F.normalize(direction, dim=-1)


# ─── TEXT EMBEDDER ───────────────────────────────────────────────────────────
class TextEmbedder(nn.Module):
    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.N_CLASSES, cfg.D_TEXT)
        nn.init.orthogonal_(self.embed.weight)   # start orthogonal — maximally separated
        self.proj  = nn.Sequential(
            nn.Linear(cfg.D_TEXT, cfg.D_TEXT),
            nn.LayerNorm(cfg.D_TEXT),
        )

    def forward(self, ids):
        return F.normalize(self.proj(self.embed(ids)), dim=-1)


# ─── FULL TSTA MODEL ─────────────────────────────────────────────────────────
class TSTA(nn.Module):
    def __init__(self, cfg: TSTAConfig):
        super().__init__()
        self.cfg         = cfg
        self.patcher     = EEGPatcher(cfg)
        self.transformer = TemporalTransformer(cfg)
        self.plta        = PLTA(cfg)
        self.traj_head   = TrajectoryHead(cfg)
        self.text_embed  = TextEmbedder(cfg)
        self.log_temp    = nn.Parameter(torch.tensor(np.log(cfg.TEMPERATURE)))

    def encode_eeg(self, x):
        tokens     = self.patcher(x)
        tokens     = self.transformer(tokens)
        plta_ctx, gates = self.plta(tokens)
        direction  = self.traj_head(tokens, plta_ctx)
        return direction, gates

    def encode_text(self, ids):
        return self.text_embed(ids)

    def forward(self, x, ids):
        eeg_dir, gates = self.encode_eeg(x)
        text_emb       = self.encode_text(ids)
        return eeg_dir, text_emb, gates

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ─── InfoNCE LOSS ─────────────────────────────────────────────────────────────
def infonce_loss(eeg, text, log_temp):
    temp   = torch.exp(log_temp).clamp(0.01, 1.0)
    sim    = torch.matmul(eeg, text.T) / temp
    labels = torch.arange(len(eeg), device=eeg.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


if __name__ == '__main__':
    cfg = TSTAConfig()
    m   = TSTA(cfg)
    print(f"TSTA model: {m.n_params():,} parameters")
    print(f"PLTA gate centers (init): {cfg.PLTA_CENTERS_S} seconds")
    x   = torch.randn(4, cfg.N_CHANNELS, cfg.N_SAMPLES)
    ids = torch.tensor([0,1,2,3])
    d, t, g = m(x, ids)
    print(f"EEG direction shape : {d.shape}")
    print(f"Text embed shape    : {t.shape}")
    print(f"PLTA gates shape    : {g.shape}")
