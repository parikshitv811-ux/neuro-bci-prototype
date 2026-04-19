"""
TSTA Project — Shared Utilities
================================
Logging, seeding, timing, and common helpers.
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
from datetime import datetime
from tsta_project.config import LOGS_DIR


# ─── LOGGING ──────────────────────────────────────────────────────────────────
def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Returns a logger that writes to console + optional file.
    Files are saved to outputs/logs/.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(os.path.join(LOGS_DIR, log_file), mode="w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─── REPRODUCIBILITY ──────────────────────────────────────────────────────────
def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── DEVICE ───────────────────────────────────────────────────────────────────
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ─── TIMING ───────────────────────────────────────────────────────────────────
class Timer:
    def __init__(self, label: str = ""):
        self.label = label
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *_):
        self.elapsed = time.time() - self._start
        if self.label:
            print(f"  ⏱  {self.label}: {self.elapsed:.1f}s")

    @property
    def elapsed_str(self):
        return f"{self.elapsed:.1f}s"


# ─── JSON SAVE ────────────────────────────────────────────────────────────────
def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── PRINT BANNER ─────────────────────────────────────────────────────────────
def banner(title: str, width: int = 65):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def section(title: str, width: int = 60):
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")
