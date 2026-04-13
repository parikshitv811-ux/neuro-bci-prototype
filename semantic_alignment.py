"""
BCI Level 3 — Semantic EEG Decoding Research Scaffold
=======================================================
Explores feasibility of aligning EEG latent representations
with text embeddings from an LLM.

Components:
  1. EEGTransformerEncoder — Transformer-based EEG feature extractor
  2. TextEmbeddingModel    — Sentence-level text encoder (simulated or real)
  3. ContrastivePairDataset — Paired (EEG epoch, text phrase) dataset
  4. EEGTextAligner        — CLIP-style contrastive alignment training
  5. SemanticSimilarityEval — Alignment quality metrics

Research context:
  - Full semantic thought-to-text is currently unsolved for non-invasive EEG.
  - This scaffold tests the *feasibility* of coarse semantic clustering:
    can EEG embeddings for "communication" tasks cluster near "email",
    while "navigation" tasks cluster near "scroll/navigate"?
  - A positive result here does NOT mean we can decode arbitrary thoughts —
    it means we may be able to decode broad categorical semantic intent.

References:
  - EEGNet: Lawhern et al. (2018)
  - CLIP contrastive alignment: Radford et al. (2021)
  - EEG+NLP alignment: Defossez et al. "Decoding speech from non-invasive
    brain recordings" (2023) — the closest real-world result.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────
# 1. EEG TRANSFORMER ENCODER
# ─────────────────────────────────────────────────────────────
class EEGTransformerEncoder(nn.Module):
    """
    Transformer encoder for EEG signals.
    Input:  (batch, n_channels, n_times)
    Output: (batch, embed_dim) — latent EEG representation

    Architecture:
    - Channel embedding: project each channel's time series to d_model
    - Positional encoding over time steps
    - N transformer encoder layers
    - CLS token pooling → final embedding

    This is closer to current SOTA than a CNN for semantic tasks,
    as it captures long-range temporal dependencies.
    """

    def __init__(self, n_channels: int = 14, n_times: int = 512,
                 d_model: int = 64, nhead: int = 4, n_layers: int = 3,
                 embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Patch the time series: split n_times into patches of size patch_size
        patch_size = 32
        n_patches = n_times // patch_size
        self.patch_size = patch_size
        self.n_patches = n_patches

        # Project each (n_channels × patch_size) patch to d_model
        self.patch_embed = nn.Linear(n_channels * patch_size, d_model)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Project to shared embedding space
        self.proj = nn.Sequential(
            nn.Linear(d_model, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) — batch of EEG epochs
        returns: (B, embed_dim) — normalized embeddings
        """
        B, C, T = x.shape
        # Create patches: (B, n_patches, C*patch_size)
        n_full = (T // self.patch_size) * self.patch_size
        x = x[:, :, :n_full]
        patches = x.reshape(B, C, -1, self.patch_size)         # (B, C, n_p, ps)
        patches = patches.permute(0, 2, 1, 3)                   # (B, n_p, C, ps)
        patches = patches.reshape(B, -1, C * self.patch_size)   # (B, n_p, C*ps)

        # Embed patches
        x = self.patch_embed(patches)                            # (B, n_p, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                          # (B, n_p+1, d_model)

        # Add positional encoding (trim/pad if needed)
        n_seq = x.shape[1]
        pos = self.pos_embed[:, :n_seq, :]
        x = x + pos

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # CLS token output
        cls_out = x[:, 0, :]                                     # (B, d_model)

        # Project + L2-normalize (for cosine similarity)
        emb = self.proj(cls_out)
        return F.normalize(emb, dim=-1)


# ─────────────────────────────────────────────────────────────
# 2. TEXT EMBEDDING MODEL  (simulated / real)
# ─────────────────────────────────────────────────────────────
class TextEmbeddingModel(nn.Module):
    """
    Maps text phrases to the same embedding space as EEG.

    In production: use a frozen sentence transformer or the
    embeddings endpoint of Claude/OpenAI and project down.

    In simulation: uses a learnable lookup table over a fixed vocabulary.
    """
    PHRASES = [
        # Communication
        "compose and send an email",
        "write a message to a contact",
        "open the email application",
        "send a quick note",
        # Navigation
        "scroll down the page",
        "scroll up to the top",
        "navigate to the next section",
        "move down through content",
        # Interaction
        "click on the button",
        "select an item",
        "press enter",
        "confirm the action",
        # System
        "open a new application",
        "launch the terminal",
        "open the browser",
        # Rest
        "do nothing",
        "idle state",
        "waiting for input",
    ]

    # Coarse semantic categories for evaluation
    CATEGORIES = {
        "communication": [0, 1, 2, 3],
        "navigation": [4, 5, 6, 7],
        "interaction": [8, 9, 10, 11],
        "system": [12, 13, 14],
        "idle": [15, 16, 17],
    }

    def __init__(self, embed_dim: int = 128, use_real_encoder: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_real = use_real_encoder
        self.vocab_size = len(self.PHRASES)

        if not use_real_encoder:
            # Learnable embeddings with category-level initialization
            # Initialize so phrases in the same category start close together
            embeddings = torch.randn(self.vocab_size, embed_dim) * 0.1
            category_offset = 0.5
            for cat_i, (cat_name, indices) in enumerate(self.CATEGORIES.items()):
                direction = torch.zeros(embed_dim)
                direction[cat_i * (embed_dim // len(self.CATEGORIES)): (cat_i+1) * (embed_dim // len(self.CATEGORIES))] = category_offset
                for idx in indices:
                    embeddings[idx] += direction
            self.text_embeddings = nn.Embedding(self.vocab_size, embed_dim,
                                                 _weight=embeddings)
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, phrase_ids: torch.Tensor) -> torch.Tensor:
        """phrase_ids: (B,) → (B, embed_dim) normalized"""
        x = self.text_embeddings(phrase_ids)
        x = self.proj(x)
        return F.normalize(x, dim=-1)

    def embed_phrase(self, phrase: str) -> torch.Tensor:
        """Embed a single phrase by finding closest in vocabulary."""
        if phrase in self.PHRASES:
            idx = self.PHRASES.index(phrase)
        else:
            # Fallback: find most similar by substring
            idx = next((i for i, p in enumerate(self.PHRASES) if phrase in p), 0)
        with torch.no_grad():
            return self(torch.tensor([idx]))


# ─────────────────────────────────────────────────────────────
# 3. PAIRED DATASET
# ─────────────────────────────────────────────────────────────
class EEGTextPairDataset(Dataset):
    """
    Paired (EEG epoch, phrase_id) dataset for contrastive training.

    In a real experiment:
    - Present the user with a phrase on screen
    - Record 2-4s of EEG while they imagine/internalize the phrase
    - Store (EEG epoch, phrase) pairs

    Here: simulate with class-frequency-matched EEG + category-matched phrases.
    """

    # Map EEG class → related text phrase indices
    EEG_TO_PHRASE = {
        0: [2, 0, 1, 3],    # open_app → communication phrases
        1: [4, 6, 7],       # scroll_down → navigation
        2: [5, 6],          # scroll_up → navigation
        3: [8, 9, 10, 11],  # click → interaction
        4: [15, 16, 17],    # idle → idle
    }
    # Corresponding EEG frequency signatures (from EEGSimulator)
    CLASS_FREQ = {0:(12,15), 1:(8,10), 2:(10,12), 3:(15,25), 4:(2,6)}

    def __init__(self, n_per_class: int = 100, sfreq: int = 256, n_channels: int = 14,
                 epoch_len: float = 2.0, noise: float = 0.3):
        self.sfreq = sfreq
        self.n_times = int(sfreq * epoch_len)
        self.data: List[Tuple[np.ndarray, int]] = []

        for cls_id, phrase_ids in self.EEG_TO_PHRASE.items():
            lo, hi = self.CLASS_FREQ[cls_id]
            for _ in range(n_per_class):
                t = np.linspace(0, epoch_len, self.n_times)
                freq = np.random.uniform(lo, hi)
                epoch = np.zeros((n_channels, self.n_times), dtype=np.float32)
                for ch in range(n_channels):
                    phase = np.random.uniform(0, 2*np.pi)
                    epoch[ch] = np.sin(2*np.pi*freq*t + phase)
                    epoch[ch] += noise * np.random.randn(self.n_times)
                phrase_id = int(np.random.choice(phrase_ids))
                self.data.append((epoch, phrase_id))

        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        epoch, phrase_id = self.data[idx]
        return torch.tensor(epoch), torch.tensor(phrase_id, dtype=torch.long)


# ─────────────────────────────────────────────────────────────
# 4. CLIP-STYLE CONTRASTIVE ALIGNER
# ─────────────────────────────────────────────────────────────
class EEGTextAligner(nn.Module):
    """
    Aligns EEG and text representations using InfoNCE contrastive loss
    (same objective as CLIP).

    Goal: EEG embeddings of "scroll down" imagination should be nearest-
    neighbor to the text embedding of navigation phrases, not communication.

    This does NOT decode free thought — it learns a mapping from
    guided-paradigm EEG to semantic categories. A more constrained but
    scientifically grounded first step.
    """

    def __init__(self, n_channels: int = 14, n_times: int = 512, embed_dim: int = 128):
        super().__init__()
        self.eeg_encoder  = EEGTransformerEncoder(n_channels, n_times, embed_dim=embed_dim)
        self.text_encoder = TextEmbeddingModel(embed_dim)
        self.logit_scale  = nn.Parameter(torch.tensor(np.log(1/0.07)))

    def forward(self, eeg: torch.Tensor, phrase_ids: torch.Tensor):
        """
        eeg:       (B, C, T)
        phrase_ids: (B,)
        Returns: loss (scalar), eeg_emb (B, D), text_emb (B, D)
        """
        eeg_emb  = self.eeg_encoder(eeg)
        text_emb = self.text_encoder(phrase_ids)

        # Scaled cosine similarity matrix
        scale = self.logit_scale.exp().clamp(max=100)
        logits = scale * (eeg_emb @ text_emb.T)           # (B, B)

        # Symmetric InfoNCE loss
        labels = torch.arange(len(eeg), device=eeg.device)
        loss_eeg  = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits.T, labels)
        loss = (loss_eeg + loss_text) / 2

        return loss, eeg_emb, text_emb

    @torch.no_grad()
    def retrieve(self, eeg: torch.Tensor, text_encoder: TextEmbeddingModel,
                 phrase_ids: torch.Tensor, top_k: int = 3) -> List[dict]:
        """
        Given an EEG epoch, retrieve the most semantically similar phrases.
        """
        eeg_emb = self.eeg_encoder(eeg)
        phrase_ids_all = torch.arange(len(TextEmbeddingModel.PHRASES))
        text_embs = text_encoder(phrase_ids_all)
        sims = (eeg_emb @ text_embs.T)  # (B, n_phrases)
        results = []
        for b in range(eeg_emb.shape[0]):
            top = sims[b].topk(top_k)
            results.append({
                "top_phrases": [TextEmbeddingModel.PHRASES[i] for i in top.indices.tolist()],
                "similarities": [round(s, 4) for s in top.values.tolist()]
            })
        return results


# ─────────────────────────────────────────────────────────────
# 5. TRAINING + EVALUATION
# ─────────────────────────────────────────────────────────────
class Level3Trainer:
    def __init__(self, model: EEGTextAligner, device: str = 'cpu'):
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=20)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for eeg, phrase_ids in loader:
            eeg = eeg.to(self.device)
            phrase_ids = phrase_ids.to(self.device)
            self.opt.zero_grad()
            loss, _, _ = self.model(eeg, phrase_ids)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total_loss += loss.item()
        self.scheduler.step()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate_retrieval(self, loader: DataLoader) -> dict:
        """
        Evaluate: for each EEG epoch, does the closest text match belong
        to the correct semantic category?
        """
        self.model.eval()
        phrase_cats = {}
        for cat, indices in TextEmbeddingModel.CATEGORIES.items():
            for idx in indices:
                phrase_cats[idx] = cat

        eeg_class_to_cat = {0:"communication", 1:"navigation", 2:"navigation",
                             3:"interaction", 4:"idle"}
        # Map phrase_id back to EEG class for ground truth
        phrase_to_eeg_cat = {}
        for eeg_cls, phrase_ids in EEGTextPairDataset.EEG_TO_PHRASE.items():
            gt_cat = eeg_class_to_cat[eeg_cls]
            for pid in phrase_ids:
                phrase_to_eeg_cat[pid] = gt_cat

        correct_top1 = 0
        correct_cat  = 0
        total = 0

        phrase_ids_all = torch.arange(len(TextEmbeddingModel.PHRASES)).to(self.device)
        all_text_embs  = self.model.text_encoder(phrase_ids_all)

        for eeg, gt_phrase_ids in loader:
            eeg = eeg.to(self.device)
            eeg_emb = self.model.eeg_encoder(eeg)
            sims = eeg_emb @ all_text_embs.T     # (B, n_phrases)
            top1_ids = sims.argmax(dim=-1).cpu().tolist()
            gt_ids   = gt_phrase_ids.tolist()

            for pred_pid, gt_pid in zip(top1_ids, gt_ids):
                total += 1
                if pred_pid == gt_pid:
                    correct_top1 += 1
                gt_cat   = phrase_cats.get(gt_pid, "unknown")
                pred_cat = phrase_cats.get(pred_pid, "unknown")
                if pred_cat == gt_cat:
                    correct_cat += 1

        return {
            "top1_accuracy":      round(correct_top1 / (total + 1e-9), 4),
            "category_accuracy":  round(correct_cat  / (total + 1e-9), 4),
            "total_samples":      total
        }


def run_level3_research(n_epochs: int = 20, batch_size: int = 32):
    """Full Level 3 research training and evaluation."""
    print("\n" + "═" * 65)
    print("  BCI LEVEL 3 — SEMANTIC ALIGNMENT RESEARCH")
    print("═" * 65)
    print("  Hypothesis: Can EEG embeddings from guided motor-imagery tasks")
    print("  align with text embeddings of semantically related phrases?")
    print("  Architecture: EEG Transformer + Text Encoder + InfoNCE loss")
    print("─" * 65 + "\n")

    torch.manual_seed(42); np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = EEGTextPairDataset(n_per_class=120)
    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model   = EEGTextAligner(n_channels=14, n_times=512, embed_dim=128)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Training samples: {n_train}  |  Validation: {n_val}")
    print(f"  Phrases in vocab: {len(TextEmbeddingModel.PHRASES)}")
    print(f"  Semantic categories: {list(TextEmbeddingModel.CATEGORIES.keys())}\n")

    trainer = Level3Trainer(model, device)

    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Top-1 Acc':>10}  {'Cat Acc':>9}")
    print("  " + "─" * 42)
    best_cat_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        loss = trainer.train_epoch(train_loader)
        if epoch % 5 == 0 or epoch == 1:
            metrics = trainer.evaluate_retrieval(val_loader)
            top1 = metrics["top1_accuracy"]
            cat  = metrics["category_accuracy"]
            flag = " ← best" if cat > best_cat_acc else ""
            best_cat_acc = max(best_cat_acc, cat)
            print(f"  {epoch:>5}  {loss:>11.4f}  {top1*100:>9.1f}%  {cat*100:>8.1f}%{flag}")

    # Final retrieval demo
    print(f"\n  Best category accuracy: {best_cat_acc*100:.1f}%")
    print(f"\n  ── Retrieval demo (3 sample EEG epochs) ──")
    model.eval()
    sample_loader = DataLoader(val_ds, batch_size=3)
    sample_eeg, sample_phrase_ids = next(iter(sample_loader))
    results = model.retrieve(sample_eeg.to(device), model.text_encoder,
                              sample_phrase_ids.to(device), top_k=3)
    for i, r in enumerate(results):
        gt = TextEmbeddingModel.PHRASES[sample_phrase_ids[i].item()]
        print(f"\n  Sample {i+1}: Ground truth → '{gt}'")
        print("    Top-3 retrieved:")
        for phrase, sim in zip(r["top_phrases"], r["similarities"]):
            match = "✓" if phrase == gt else "  "
            print(f"    {match} '{phrase}'  (sim={sim:.3f})")

    print(f"\n{'═'*65}")
    print("  RESEARCH NOTES")
    print("─" * 65)
    print("  • Category accuracy > top-1 accuracy is the expected result.")
    print("  • Even coarse semantic clustering (communication vs navigation)")
    print("    would be a significant result for non-invasive EEG.")
    print("  • Next steps: real EEG data, larger transformer, EEG foundation")
    print("    model pre-training (e.g., BENDR, LaBraM), contrastive fine-tuning.")
    print("  • Real feasibility bar: ~60% category accuracy on held-out subjects.")
    print(f"{'═'*65}\n")

    return {"best_category_accuracy": best_cat_acc, "n_params": n_params}


if __name__ == "__main__":
    results = run_level3_research(n_epochs=20, batch_size=32)
