"""
TSTA Project — Centralized Configuration
=========================================
All hyperparameters and paths in one place.
"""

import os

# ─── PROJECT PATHS ────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR   = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR    = os.path.join(OUTPUTS_DIR, "models")
FIGURES_DIR   = os.path.join(OUTPUTS_DIR, "figures")
LOGS_DIR      = os.path.join(OUTPUTS_DIR, "logs")

for _d in [MODELS_DIR, FIGURES_DIR, LOGS_DIR]:
    os.makedirs(_d, exist_ok=True)


# ─── MODEL CONFIGURATION ──────────────────────────────────────────────────────
class TSTAConfig:
    """
    Configuration for the TSTA model and training pipeline.
    Call update_from_dataset() after loading data to adapt to its shape.
    """

    # EEG data defaults (overridden by dataset)
    N_CHANNELS  = 64
    SFREQ       = 160
    N_SAMPLES   = 320          # 2s × 160Hz

    # Patching
    PATCH_LEN   = 40           # ~250ms at 160Hz
    PATCH_STEP  = 20           # ~125ms stride
    N_PATCHES   = (N_SAMPLES - PATCH_LEN) // PATCH_STEP + 1  # 15

    # Model dims
    D_MODEL     = 128
    N_HEADS     = 4
    N_LAYERS    = 3
    D_TEXT      = 128
    DROPOUT     = 0.15

    # PLTA gate centers (seconds) — P2, N2/P3, late component
    PLTA_CENTERS_S = [0.20, 0.30, 0.50]
    PLTA_WIDTH_S   = 0.08

    # Training
    BATCH_SIZE  = 32
    LR          = 3e-4
    EPOCHS      = 40
    TEMPERATURE = 0.07

    # Semantic classes
    N_CLASSES   = 5
    INTENTS     = ['communication', 'navigation', 'action', 'selection', 'idle']

    # EEG epoch length (seconds)
    EPOCH_LEN    = 2.0

    # Noise robustness sigma levels
    NOISE_SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]

    # Evaluation targets
    TARGET_WITHIN_SDAS = 0.4
    TARGET_CROSS_SDAS  = 0.25
    TARGET_TOP1        = 0.6

    def update_from_dataset(self, n_channels: int, sfreq: float, n_samples: int):
        """Adapt config dimensions to match the loaded dataset."""
        self.N_CHANNELS = n_channels
        self.SFREQ      = sfreq
        self.N_SAMPLES  = n_samples
        self.EPOCH_LEN  = n_samples / sfreq
        self.PATCH_LEN  = max(16, int(0.25 * sfreq))
        self.PATCH_STEP = max(8,  int(0.125 * sfreq))
        self.N_PATCHES  = (n_samples - self.PATCH_LEN) // self.PATCH_STEP + 1

    def __repr__(self):
        return (f"TSTAConfig(C={self.N_CHANNELS}, sfreq={self.SFREQ}Hz, "
                f"T={self.N_SAMPLES}, patches={self.N_PATCHES}, "
                f"D={self.D_MODEL}, L={self.N_LAYERS})")
