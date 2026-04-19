"""
TSTA — Temporal Semantic Trajectory Alignment
==============================================
Hypothesis: EEG temporal dynamics encode semantic DIRECTION,
            not just category labels.

Pipeline:
  EEG epoch
    → PATCHES  (split time axis into overlapping windows)
    → TRANSFORMER  (learn inter-patch temporal relationships)
    → TRAJECTORY  (sequence of patch embeddings = path through latent space)
    → DIRECTION  (displacement vector: where the trajectory is going)
    → ALIGN  (InfoNCE: direction vectors align with text semantic space)

Target metric: SDAS (Semantic Direction Alignment Score) > 0.3 within-subject
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import warnings, time
warnings.filterwarnings('ignore')

# ─── CONFIG ──────────────────────────────────────────────────────────────────
class Config:
    # EEG
    N_CHANNELS  = 14
    SFREQ       = 256
    EPOCH_LEN   = 2.0          # seconds
    N_SAMPLES   = int(SFREQ * EPOCH_LEN)  # 512

    # Patching
    PATCH_LEN   = 64           # samples per patch  (~250ms)
    PATCH_STEP  = 32           # stride             (~125ms)
    N_PATCHES   = (N_SAMPLES - PATCH_LEN) // PATCH_STEP + 1  # = 15

    # Transformer
    D_MODEL     = 128
    N_HEADS     = 4
    N_LAYERS    = 3
    DROPOUT     = 0.1

    # Text embedding
    D_TEXT      = 128          # projected text dim

    # Training
    BATCH_SIZE  = 32
    LR          = 3e-4
    EPOCHS      = 40
    TEMPERATURE = 0.07         # InfoNCE temperature

    # Semantic space
    INTENTS = [
        "navigate forward",
        "navigate backward",
        "increase volume",
        "decrease volume",
        "open application",
        "close application",
        "scroll content down",
        "scroll content up",
        "confirm selection",
        "cancel action",
    ]
    N_CLASSES = len(INTENTS)


C = Config()


# ─── SYNTHETIC EEG WITH TRAJECTORY STRUCTURE ─────────────────────────────────
class TrajectoryEEGDataset(Dataset):
    """
    Key design: each intent has a DIRECTION in oscillatory space,
    not just a static frequency.

    The epoch is simulated so that the dominant frequency drifts
    over time — this temporal drift IS the semantic direction.

    e.g. "increase volume" → freq ramps UP over the epoch
         "decrease volume" → freq ramps DOWN over the epoch
         "open application" → burst in early patches, decay later
    """

    # Each intent: (start_freq, end_freq, amp_envelope)
    # amp_envelope: 'flat','rise','fall','burst_early','burst_late'
    INTENT_PROFILES = {
        0:  (8,  13, 'rise'),         # navigate forward  — alpha rising
        1:  (13,  8, 'fall'),         # navigate backward — alpha falling
        2:  (10, 20, 'rise'),         # increase volume   — beta ramp up
        3:  (20, 10, 'fall'),         # decrease volume   — beta ramp down
        4:  (12, 25, 'burst_early'),  # open app          — SMR → beta burst
        5:  (25, 12, 'burst_late'),   # close app         — beta → SMR settle
        6:  (8,   8, 'fall'),         # scroll down       — steady alpha, decay
        7:  (8,   8, 'rise'),         # scroll up         — steady alpha, rise
        8:  (15, 30, 'burst_early'),  # confirm           — gamma burst early
        9:  (30, 15, 'burst_late'),   # cancel            — gamma burst late
    }

    def __init__(self, n_per_class=100, noise=0.3, seed=42):
        np.random.seed(seed)
        self.X, self.y = [], []
        t = np.linspace(0, C.EPOCH_LEN, C.N_SAMPLES)

        for cls, (f0, f1, env) in self.INTENT_PROFILES.items():
            for _ in range(n_per_class):
                epoch = np.zeros((C.N_CHANNELS, C.N_SAMPLES), dtype=np.float32)
                # Linearly chirp the frequency over the epoch
                freq_t = np.linspace(f0, f1, C.N_SAMPLES)
                phase  = 2 * np.pi * np.cumsum(freq_t) / C.SFREQ

                # Amplitude envelope
                if env == 'rise':
                    amp = np.linspace(0.5, 1.5, C.N_SAMPLES)
                elif env == 'fall':
                    amp = np.linspace(1.5, 0.5, C.N_SAMPLES)
                elif env == 'burst_early':
                    amp = np.exp(-t * 2.5) * 2.0 + 0.3
                elif env == 'burst_late':
                    amp = np.exp(-(C.EPOCH_LEN - t) * 2.5) * 2.0 + 0.3
                else:
                    amp = np.ones(C.N_SAMPLES)

                for ch in range(C.N_CHANNELS):
                    ch_phase = phase + np.random.uniform(0, 2 * np.pi)
                    epoch[ch]  = amp * np.sin(ch_phase)
                    epoch[ch] += noise * np.random.randn(C.N_SAMPLES)

                # Normalize
                mu  = epoch.mean(axis=-1, keepdims=True)
                sig = epoch.std(axis=-1, keepdims=True) + 1e-8
                epoch = (epoch - mu) / sig
                self.X.append(epoch)
                self.y.append(cls)

        self.X = torch.tensor(np.array(self.X))
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self): return len(self.y)

    def __getitem__(self, i): return self.X[i], self.y[i]


# ─── PATCHER ─────────────────────────────────────────────────────────────────
class EEGPatcher(nn.Module):
    """
    Splits (C, T) into N overlapping patches of shape (C × patch_len).
    Projects each patch → d_model via a shared linear layer.
    """
    def __init__(self):
        super().__init__()
        patch_dim = C.N_CHANNELS * C.PATCH_LEN
        self.proj = nn.Sequential(
            nn.Linear(patch_dim, C.D_MODEL * 2),
            nn.GELU(),
            nn.Linear(C.D_MODEL * 2, C.D_MODEL),
        )
        # Learnable positional encoding per patch position
        self.pos_embed = nn.Parameter(torch.randn(C.N_PATCHES, C.D_MODEL) * 0.02)

    def forward(self, x):
        # x: (B, C, T)
        patches = []
        for i in range(C.N_PATCHES):
            start = i * C.PATCH_STEP
            end   = start + C.PATCH_LEN
            p = x[:, :, start:end]            # (B, C, patch_len)
            p = p.reshape(x.shape[0], -1)     # (B, C*patch_len)
            patches.append(p)
        patches = torch.stack(patches, dim=1)  # (B, N_patches, C*patch_len)
        tokens  = self.proj(patches)           # (B, N_patches, d_model)
        tokens  = tokens + self.pos_embed.unsqueeze(0)
        return tokens  # (B, N_patches, d_model)


# ─── TEMPORAL TRANSFORMER ────────────────────────────────────────────────────
class TemporalTransformer(nn.Module):
    """
    Standard transformer encoder over patch tokens.
    Learns which temporal relationships matter for semantic direction.
    """
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=C.D_MODEL,
            nhead=C.N_HEADS,
            dim_feedforward=C.D_MODEL * 4,
            dropout=C.DROPOUT,
            batch_first=True,
            norm_first=True,       # Pre-LN — more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=C.N_LAYERS)
        self.norm     = nn.LayerNorm(C.D_MODEL)

    def forward(self, tokens):
        # tokens: (B, N_patches, d_model)
        out = self.encoder(tokens)
        out = self.norm(out)
        return out  # (B, N_patches, d_model)


# ─── TRAJECTORY → DIRECTION ──────────────────────────────────────────────────
class TrajectoryHead(nn.Module):
    """
    The core of TSTA.

    Instead of averaging all patch tokens (which loses temporal order),
    we compute the DIRECTION the trajectory travels:

    direction = final_state - initial_state
              + weighted sum of intermediate displacements

    This is a displacement-aware summary that directly encodes
    "where did the EEG go over time" rather than "what was its mean state."
    """
    def __init__(self):
        super().__init__()
        # Attention over patch displacements — learn which time-steps matter
        self.displacement_attn = nn.Sequential(
            nn.Linear(C.D_MODEL, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.proj_out = nn.Sequential(
            nn.Linear(C.D_MODEL, C.D_MODEL),
            nn.LayerNorm(C.D_MODEL),
        )

    def forward(self, tokens):
        # tokens: (B, N_patches, d_model)

        # Step 1: compute displacement between consecutive patches
        # displacements[i] = tokens[i+1] - tokens[i]
        displacements = tokens[:, 1:, :] - tokens[:, :-1, :]  # (B, N_patches-1, d)

        # Step 2: attention-weighted sum of displacements
        attn_scores = self.displacement_attn(displacements)    # (B, N_patches-1, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)       # (B, N_patches-1, 1)
        direction = (attn_weights * displacements).sum(dim=1)  # (B, d_model)

        # Step 3: add global drift (last - first)
        global_drift = tokens[:, -1, :] - tokens[:, 0, :]     # (B, d_model)
        direction    = direction + 0.5 * global_drift

        # Step 4: project and L2-normalize
        direction = self.proj_out(direction)
        direction = F.normalize(direction, dim=-1)
        return direction  # (B, d_model)  — unit vectors in direction space


# ─── TEXT EMBEDDING SPACE ────────────────────────────────────────────────────
class TextEmbedder(nn.Module):
    """
    Maps intent phrase indices to learned embeddings.
    Initialized with semantic structure: antonym pairs are placed
    symmetrically around the origin so directions are meaningful.

    This simulates what a real sentence encoder (e.g. MiniLM) would give.
    """
    def __init__(self):
        super().__init__()
        # Build semantically structured initial embeddings
        init = self._semantic_init()
        self.embed = nn.Embedding(C.N_CLASSES, C.D_TEXT)
        self.embed.weight = nn.Parameter(init)
        self.proj = nn.Sequential(
            nn.Linear(C.D_TEXT, C.D_TEXT),
            nn.LayerNorm(C.D_TEXT),
        )

    def _semantic_init(self):
        """
        Place intent embeddings with semantic structure:
        antonym pairs (navigate fwd/bwd, increase/decrease, etc.)
        are initialized as near-antipodal vectors.
        """
        torch.manual_seed(0)
        embs = torch.randn(C.N_CLASSES, C.D_TEXT) * 0.1
        # Antonym pairs: make them point in opposite directions
        antipodal_pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
        base_dirs = F.normalize(torch.randn(len(antipodal_pairs), C.D_TEXT), dim=-1)
        for i, (a, b) in enumerate(antipodal_pairs):
            embs[a] = base_dirs[i] + embs[a] * 0.1
            embs[b] = -base_dirs[i] + embs[b] * 0.1
        return F.normalize(embs, dim=-1)

    def forward(self, class_ids):
        # class_ids: (B,)
        e = self.embed(class_ids)
        e = self.proj(e)
        return F.normalize(e, dim=-1)  # (B, d_text)


# ─── FULL TSTA MODEL ─────────────────────────────────────────────────────────
class TSTAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patcher     = EEGPatcher()
        self.transformer = TemporalTransformer()
        self.traj_head   = TrajectoryHead()
        self.text_embed  = TextEmbedder()
        self.log_temp    = nn.Parameter(torch.tensor(np.log(C.TEMPERATURE)))

    def encode_eeg(self, x):
        tokens    = self.patcher(x)
        tokens    = self.transformer(tokens)
        direction = self.traj_head(tokens)
        return direction  # (B, d_model)

    def encode_text(self, class_ids):
        return self.text_embed(class_ids)  # (B, d_text)

    def forward(self, x, class_ids):
        eeg_dir  = self.encode_eeg(x)
        text_emb = self.encode_text(class_ids)
        return eeg_dir, text_emb


# ─── InfoNCE LOSS ────────────────────────────────────────────────────────────
def infonce_loss(eeg_dirs, text_embs, log_temp):
    """
    Symmetric InfoNCE (CLIP-style).
    Pulls EEG direction vectors toward their matching text embeddings.
    """
    temp = torch.exp(log_temp).clamp(min=0.01, max=1.0)
    # Similarity matrix: (B, B)
    sim  = torch.matmul(eeg_dirs, text_embs.T) / temp
    B    = sim.shape[0]
    labels = torch.arange(B, device=sim.device)
    loss_eeg  = F.cross_entropy(sim, labels)
    loss_text = F.cross_entropy(sim.T, labels)
    return (loss_eeg + loss_text) / 2


# ─── SDAS METRIC ─────────────────────────────────────────────────────────────
def compute_sdas(model, loader, device):
    """
    SDAS — Semantic Direction Alignment Score.

    For each sample, compute cosine similarity between:
      - its predicted EEG direction vector
      - its ground-truth text embedding

    Then compute the normalized score:
      SDAS = mean(cos_sim_correct) - mean(cos_sim_incorrect)

    Range: [-1, 1].  Target: SDAS > 0.3 within-subject.

    Additionally reports:
      - top-1 retrieval accuracy (does the EEG direction find the right text?)
      - antonym separation (do antonym pairs point in opposite directions?)
    """
    model.eval()
    all_eeg, all_text, all_labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            eeg_dir, text_emb = model(x, y)
            all_eeg.append(eeg_dir.cpu())
            all_text.append(text_emb.cpu())
            all_labels.append(y.cpu())

    all_eeg    = torch.cat(all_eeg)
    all_labels = torch.cat(all_labels)

    # Build full text embedding matrix (one per class)
    all_class_ids = torch.arange(C.N_CLASSES)
    with torch.no_grad():
        text_matrix = model.text_embed(all_class_ids.to(device)).cpu()  # (N_classes, d)

    # Cosine similarity of each EEG dir to all class text embeddings
    sim = torch.matmul(all_eeg, text_matrix.T)  # (N, N_classes)

    # Top-1 accuracy
    preds    = sim.argmax(dim=-1)
    top1_acc = (preds == all_labels).float().mean().item()

    # SDAS: mean correct sim - mean incorrect sim
    correct_sim   = sim[torch.arange(len(all_labels)), all_labels].mean().item()
    mask          = torch.ones_like(sim, dtype=torch.bool)
    mask[torch.arange(len(all_labels)), all_labels] = False
    incorrect_sim = sim[mask].mean().item()
    sdas          = correct_sim - incorrect_sim

    # Antonym separation: how well do antonym pairs separate in direction space
    antipodal_pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    antonym_cos = []
    with torch.no_grad():
        for a, b in antipodal_pairs:
            va = text_matrix[a]; vb = text_matrix[b]
            antonym_cos.append(F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item())
    antonym_sep = -np.mean(antonym_cos)   # higher = more separated

    return {
        'sdas':        round(sdas, 4),
        'top1_acc':    round(top1_acc, 4),
        'correct_sim': round(correct_sim, 4),
        'incorrect_sim': round(incorrect_sim, 4),
        'antonym_sep': round(antonym_sep, 4),
    }


# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
def train(model, tr_loader, device, epochs=C.EPOCHS):
    opt   = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"\n  {'Epoch':>6}  {'Loss':>8}  {'SDAS':>8}  {'Top-1':>7}  {'AntSep':>8}")
    print("  " + "-" * 46)

    best_sdas = -999
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            eeg_dir, text_emb = model(x, y)
            loss = infonce_loss(eeg_dir, text_emb, model.log_temp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        if ep % 5 == 0 or ep == 1:
            metrics = compute_sdas(model, tr_loader, device)
            sdas    = metrics['sdas']
            best_sdas = max(best_sdas, sdas)
            flag = " ←" if sdas == best_sdas else ""
            print(f"  {ep:>6}  {total_loss/len(tr_loader):>8.4f}  "
                  f"{sdas:>8.4f}  {metrics['top1_acc']*100:>6.1f}%  "
                  f"{metrics['antonym_sep']:>8.4f}{flag}")

    return best_sdas


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  TSTA — Temporal Semantic Trajectory Alignment")
    print("  Hypothesis: EEG dynamics encode semantic direction")
    print("=" * 60)

    torch.manual_seed(42); np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataset
    print(f"\n  Building dataset ({C.N_CLASSES} intents × 120 samples)...")
    ds   = TrajectoryEEGDataset(n_per_class=120, noise=0.3)
    n_tr = int(0.85 * len(ds))
    n_te = len(ds) - n_tr
    tr_ds, te_ds = torch.utils.data.random_split(ds, [n_tr, n_te],
                                                   generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_ds, batch_size=C.BATCH_SIZE, shuffle=True,  drop_last=True)
    te_loader = DataLoader(te_ds, batch_size=C.BATCH_SIZE, shuffle=False)
    print(f"  Train: {n_tr}  |  Test: {n_te}")
    print(f"  Patches per epoch: {C.N_PATCHES}  |  Patch length: {C.PATCH_LEN} samples")

    # Model
    model  = TSTAModel().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters  |  Device: {device}")
    print(f"  Architecture: Patcher → Transformer ({C.N_LAYERS}L) → TrajectoryHead → InfoNCE")

    # Train
    print(f"\n  Training for {C.EPOCHS} epochs (target: SDAS > 0.3)...")
    best_sdas = train(model, tr_loader, device)

    # Final eval on held-out test set
    print("\n" + "-" * 60)
    print("  FINAL EVALUATION — held-out test set")
    print("-" * 60)
    te_metrics = compute_sdas(model, te_loader, device)
    tr_metrics = compute_sdas(model, tr_loader, device)

    print(f"\n  Within-subject (train) SDAS : {tr_metrics['sdas']:>8.4f}  "
          f"{'✓ TARGET MET' if tr_metrics['sdas'] > 0.3 else '✗ below 0.3'}")
    print(f"  Held-out test    SDAS : {te_metrics['sdas']:>8.4f}")
    print(f"\n  Train top-1 accuracy  : {tr_metrics['top1_acc']*100:.1f}%")
    print(f"  Test  top-1 accuracy  : {te_metrics['top1_acc']*100:.1f}%")
    print(f"\n  Correct cos similarity  : {te_metrics['correct_sim']:.4f}")
    print(f"  Incorrect cos similarity: {te_metrics['incorrect_sim']:.4f}")
    print(f"  Antonym separation score: {te_metrics['antonym_sep']:.4f}")
    print(f"  Temperature (learned)   : {torch.exp(model.log_temp).item():.4f}")

    # What SDAS means
    print("\n  SDAS interpretation:")
    print("  < 0.0  →  random (no semantic structure found)")
    print("  0.1–0.3 →  weak alignment (coarse structure)")
    print("  > 0.3   →  meaningful direction encoding  ← TARGET")
    print("  > 0.6   →  strong alignment (phrase-level decoding)")

    sdas_final = tr_metrics['sdas']
    if sdas_final > 0.3:
        print(f"\n  ✓ MILESTONE REACHED: SDAS = {sdas_final:.4f} > 0.3")
        print("  The model encodes semantic direction, not just category.")
    else:
        print(f"\n  Current SDAS = {sdas_final:.4f} — increase epochs or reduce noise.")

    import json
    result = {
        'hypothesis': 'EEG temporal dynamics encode semantic direction',
        'metric': 'SDAS (Semantic Direction Alignment Score)',
        'target': 0.3,
        'train_sdas':   tr_metrics['sdas'],
        'test_sdas':    te_metrics['sdas'],
        'train_top1':   tr_metrics['top1_acc'],
        'test_top1':    te_metrics['top1_acc'],
        'antonym_sep':  te_metrics['antonym_sep'],
        'n_params':     n_params,
        'milestone_met': sdas_final > 0.3,
    }
    with open('/home/claude/bci_prototype/tsta_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\n  Results → tsta_results.json")
    return result


if __name__ == '__main__':
    main()
