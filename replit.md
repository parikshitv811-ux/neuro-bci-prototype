# TSTA — Temporal Semantic Trajectory Alignment

## Overview
Production-grade EEG research pipeline proving that EEG temporal dynamics encode semantic intent as trajectory *direction* in latent space. Supports both 64-channel synthetic EEG and real PhysioNet EEGMMIDB data.

## Architecture

### Web Interface
- `app.py` — Flask web server on port 5000
- `templates/index.html` — Pipeline control UI with live terminal, results, and figure display

### `tsta_project/` — Main Research Package
```
tsta_project/
├── config.py                  # TSTAConfig (all hyperparameters + output paths)
├── utils.py                   # Logging, seeding, timing helpers
├── data/
│   ├── preprocess.py          # Shared: bandpass + notch + baseline + z-score
│   ├── synthetic/
│   │   ├── generator.py       # SyntheticEEGGenerator (64ch, 160Hz, 5 subjects)
│   │   └── profiles.py        # Category EEG signal profiles
│   └── real/
│       └── physionet_loader.py # PhysioNet EEGMMIDB loader with synthetic fallback
├── model/
│   ├── patcher.py             # EEGPatcher (overlapping patch tokens)
│   ├── transformer.py         # Pre-LN TransformerEncoder
│   ├── plta.py                # Phase-Locked Temporal Attention (P2/N2/P3 gates)
│   ├── trajectory.py          # TrajectoryHead (displacement-based direction)
│   └── tsta_model.py          # Full TSTA model + InfoNCE loss
├── training/
│   ├── trainer.py             # TSTATrainer (AdamW + CosineAnnealingLR)
│   ├── metrics.py             # SDAS, Top-1, trajectory consistency, noise robustness
│   └── eval.py                # Within-subject, cross-subject LOO, ablation
├── viz/
│   ├── tsne.py                # t-SNE embedding plots
│   ├── trajectory_plot.py     # PCA compass + patch trajectory
│   ├── plta_viz.py            # PLTA gate profiles + text cosine heatmap
│   └── dashboard.py           # Full 8-panel validation figure
├── scripts/
│   ├── debug.py               # Sanity checks (run first!)
│   ├── run_synthetic.py       # Phase 0+1: synthetic baseline
│   ├── run_real.py            # Phase 2+3: PhysioNet / fallback
│   └── run_full_pipeline.py   # All phases end-to-end
└── outputs/
    ├── models/                # Saved .pt checkpoints
    ├── figures/               # PNG dashboards
    └── logs/                  # JSON results
```

## Running the Project
- Workflow: `python3 app.py` on port 5000
- Via UI: click pipeline buttons in the web interface
- Via CLI: `python -m tsta_project.scripts.run_synthetic`

## Key Metric: SDAS
**Semantic Direction Alignment Score** = mean(cos_sim_correct) − mean(cos_sim_incorrect)
- Within-subject target: SDAS > 0.4
- Cross-subject target: SDAS > 0.25
- Top-1 accuracy target: > 60%

## Dependencies
Python: flask, numpy, scipy, scikit-learn, matplotlib, torch

## Semantic Categories
5 classes: communication (8–12Hz), navigation (12–18Hz), action (15–25Hz), selection (10–15Hz), idle (6–9Hz)

## Deployment
Configured for autoscale deployment running `python3 app.py`.
