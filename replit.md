# BCI Prototype — Brain-Computer Interface System

## Overview
A multi-level Python research platform for decoding EEG signals into digital commands. Features synthetic EEG data generation, an EEGNet CNN classifier, and a Flask web interface for running and visualizing the pipeline.

## Architecture
- **Frontend**: Flask web app (`app.py`) served on port 5000
- **Core Pipeline**: `bci_core.py` — EEG simulation, preprocessing, EEGNet training, real-time inference
- **Level 2**: `bci_level2_pipeline.py` — Claude AI agent, hardware layer, execution engine
- **Level 3 (TSTA)**: `run_all.py`, `tsta_core.py` — contrastive semantic alignment research
- **Visualization**: `visualize.py`, `bci_dashboard.png`

## Running the Project
The main workflow starts the Flask server:
```
python3 app.py
```
This serves the web UI on port 5000. From the UI you can run the core pipeline, benchmark, or TSTA research scripts.

## Key Files
- `app.py` — Flask web server (entry point)
- `templates/index.html` — Web UI
- `bci_core.py` — Core EEG pipeline: simulator, EEGNet CNN, trainer, inference engine
- `bci_level2_pipeline.py` — Level 2 orchestrator (Claude agent, hardware, execution)
- `tsta_core.py` — Temporal Semantic Trajectory Alignment model
- `run_all.py` — Master runner for TSTA research pipeline
- `claude_agent.py` — Anthropic Claude API integration
- `hardware_layer.py` — EEG device drivers (BrainFlow, simulation)
- `execution_engine.py` — OS-level command execution (PyAutoGUI)
- `benchmark.py` — Latency benchmarking
- `visualize.py` — Dashboard plot generation

## Dependencies
Python packages: flask, numpy, scipy, scikit-learn, matplotlib, torch

## Deployment
Configured for autoscale deployment running `python3 app.py`.
