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

## Analysis Package — `tsta_project/analysis/`
```
analysis/
├── __init__.py
├── direction_invariance.py   # P1: Cross-subject direction CDAS
├── amplitude_invariance.py   # P2: Amplitude scaling AIS
├── domain_shift.py           # P3: Noise/freq-shift/dropout domain transfer
├── temporal_smoothness.py    # P4: Patch-level trajectory curvature & smoothness
├── geometric_structure.py    # P5: Angular separability, k-Means ARI, PCA
├── time_reversal.py          # P6: Causality via time-reversed EEG (TRE)
├── partial_signal.py         # P7: Early intent detectability at 25/50/75%
├── failure_modes.py          # P8: Confusion matrix, per-class precision/recall
├── advanced_viz.py           # P9: 8 publication-quality figures
└── research_report.py        # Structured claim verdict report
```

### 5 Deep Claims Tested
| Claim | Description | Metric |
|-------|-------------|--------|
| C1 | Direction ≠ amplitude | AIS (random scale) > 0.85 |
| C2 | Cross-subject consistency | CDAS > 0.05 |
| C3 | Domain shift robustness | DT-SDAS > 0.5×target |
| C4 | Temporal causality | TRE > 0.3 |
| C5 | Geometric structure | k-Means ARI > 0.3, inter-class angle > 10° |

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
