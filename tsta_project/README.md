# TSTA Project — Temporal Semantic Trajectory Alignment

**Hypothesis:** EEG temporal dynamics encode semantic intent as trajectory *direction* in latent space.

---

## Project Structure

```
tsta_project/
├── data/
│   ├── synthetic/
│   │   ├── generator.py      # High-fidelity synthetic EEG (64ch, 160Hz)
│   │   └── profiles.py       # Category signal profiles
│   ├── real/
│   │   └── physionet_loader.py # PhysioNet EEGMMIDB loader + fallback
│   └── preprocess.py          # Shared bandpass + notch + baseline + z-score
│
├── model/
│   ├── patcher.py             # EEG → overlapping patch tokens
│   ├── transformer.py         # Pre-LN TransformerEncoder
│   ├── plta.py                # Phase-Locked Temporal Attention
│   ├── trajectory.py          # Displacement-based direction head
│   └── tsta_model.py          # Full TSTA + InfoNCE loss
│
├── training/
│   ├── trainer.py             # TSTATrainer (AdamW + cosine schedule)
│   ├── metrics.py             # SDAS, Top-1, trajectory consistency
│   └── eval.py                # Within-subject, cross-subject, noise, ablation
│
├── viz/
│   ├── tsne.py                # t-SNE embedding plot
│   ├── trajectory_plot.py     # PCA compass + patch trajectory
│   ├── plta_viz.py            # PLTA gate profiles + text cosine heatmap
│   └── dashboard.py           # Full 8-panel validation figure
│
├── scripts/
│   ├── debug.py               # Sanity checks (run first)
│   ├── run_synthetic.py       # Phase 0+1: synthetic baseline
│   ├── run_real.py            # Phase 2+3: PhysioNet / fallback
│   └── run_full_pipeline.py   # All phases end-to-end
│
├── outputs/
│   ├── models/                # Saved .pt checkpoints
│   ├── figures/               # PNG dashboards
│   └── logs/                  # JSON results
│
├── config.py                  # All hyperparameters + paths
└── utils.py                   # Logging, seeding, timing
```

---

## Quick Start

```bash
# 1. Sanity check (always run first)
python -m tsta_project.scripts.debug

# 2. Synthetic baseline
python -m tsta_project.scripts.run_synthetic

# 3. Real EEG (PhysioNet with fallback to synthetic)
python -m tsta_project.scripts.run_real

# 4. Full pipeline (all phases)
python -m tsta_project.scripts.run_full_pipeline
```

---

## Key Metric: SDAS

**Semantic Direction Alignment Score**

```
SDAS = mean(cos_sim_correct) − mean(cos_sim_incorrect)
```

| Range     | Interpretation                          |
|-----------|----------------------------------------|
| < 0       | Random (no semantic structure)          |
| 0.1–0.3   | Weak alignment (coarse structure)       |
| > 0.4     | ✓ Within-subject target                 |
| > 0.25    | ✓ Cross-subject target                  |
| > 0.6     | Strong phrase-level decoding            |

---

## Architecture

```
EEG (B, C, T)
    → EEGPatcher        → patch tokens (B, N, D)
    → TemporalTransformer → contextualized tokens (B, N, D)
    → PLTA              → temporal context (B, D)  [P2/N2/P3 gates]
    → TrajectoryHead    → direction vector (B, D)  [displacement-based]
    → L2 normalize
    ↕  InfoNCE Loss
    TextEmbedder        → class embeddings (B, D)
```

---

## Semantic Categories

| ID | Label          | EEG Band   | ERP     |
|----|----------------|------------|---------|
| 0  | communication  | 8–12 Hz    | 300ms   |
| 1  | navigation     | 12–18 Hz   | 250ms   |
| 2  | action         | 15–25 Hz   | 200ms   |
| 3  | selection      | 10–15 Hz   | 350ms   |
| 4  | idle           | 6–9 Hz     | —       |
