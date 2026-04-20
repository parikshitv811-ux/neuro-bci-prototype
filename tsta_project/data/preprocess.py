"""
TSTA Project — Shared Preprocessing Pipeline
=============================================
Bandpass 1-40Hz → Notch 50Hz → Baseline correction → Per-channel z-score.
+ Subject-invariant normalization to remove subject identity signals.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from scipy.signal import butter, filtfilt, iirnotch


@dataclass
class EEGDataset:
    """Unified dataset container for both synthetic and real EEG data."""
    X:          np.ndarray     # (N, C, T)  float32
    y:          np.ndarray     # (N,)       int
    subjects:   np.ndarray     # (N,)       int subject IDs
    sfreq:      float
    ch_names:   List[str]
    categories: dict
    n_subjects: int
    source:     str            # 'synthetic' | 'physionet'

    @property
    def n_epochs(self):
        return len(self.y)

    @property
    def n_channels(self):
        return self.X.shape[1]

    @property
    def n_samples(self):
        return self.X.shape[2]

    def summary(self) -> str:
        cls_counts = {int(k): int(v)
                      for k, v in zip(*np.unique(self.y, return_counts=True))}
        return (
            f"EEGDataset(source={self.source!r}, "
            f"shape={self.X.shape}, "
            f"subjects={self.n_subjects}, "
            f"sfreq={self.sfreq}Hz, "
            f"class_counts={cls_counts})"
        )


class Preprocessor:
    """
    EEG preprocessing with optional subject-invariant normalization.

    Steps (standard):
      1. Bandpass filter  1-40 Hz
      2. Notch filter     50 Hz
      3. Baseline correction (subtract mean of first 100ms)
      4. Per-channel z-score normalization

    Steps (subject-invariant, applied after standard):
      5. Temporal energy normalization: X /= ||X||_time
      6. Channel-wise global normalization across all subjects
      7. Per-subject mean subtraction (removes subject bias)
    """

    def __init__(self, sfreq: float = 160, notch_hz: float = 50.0):
        self.sfreq    = sfreq
        self.notch_hz = notch_hz
        nyq           = sfreq / 2.0

        self.b_bp, self.a_bp = butter(4, [1.0 / nyq, 40.0 / nyq], btype="band")
        self.b_n,  self.a_n  = iirnotch(notch_hz, Q=30, fs=sfreq)

    def process(self, epoch: np.ndarray) -> np.ndarray:
        """
        Process a single epoch.

        Args:
            epoch: (C, T) float array

        Returns:
            Preprocessed epoch (C, T) float32
        """
        ep = filtfilt(self.b_bp, self.a_bp, epoch, axis=-1)
        ep = filtfilt(self.b_n,  self.a_n,  ep,    axis=-1)

        n_baseline = max(1, int(0.1 * self.sfreq))
        baseline   = ep[:, :n_baseline].mean(axis=-1, keepdims=True)
        ep         = ep - baseline

        mu  = ep.mean(axis=-1, keepdims=True)
        sig = ep.std(axis=-1, keepdims=True) + 1e-8
        return ((ep - mu) / sig).astype(np.float32)

    def process_dataset(self, ds: "EEGDataset",
                        verbose:            bool = True,
                        subject_invariant:  bool = True) -> "EEGDataset":
        """
        Apply preprocessing + optional subject-invariant normalization.

        Args:
            ds:                Input dataset
            verbose:           Print progress
            subject_invariant: Apply steps 5-7 for cross-subject alignment
        """
        if verbose:
            print(f"  [Preprocess] {len(ds.y)} epochs | "
                  f"bandpass 1-40Hz + notch {self.notch_hz}Hz + baseline...")
        X_proc = np.array([self.process(ep) for ep in ds.X], dtype=np.float32)

        if subject_invariant:
            X_proc = self._subject_invariant(X_proc, ds.subjects, verbose=verbose)

        return EEGDataset(
            X=X_proc,
            y=ds.y,
            subjects=ds.subjects,
            sfreq=ds.sfreq,
            ch_names=ds.ch_names,
            categories=ds.categories,
            n_subjects=ds.n_subjects,
            source=ds.source,
        )

    def _subject_invariant(self, X: np.ndarray, subjects: np.ndarray,
                           verbose: bool = True) -> np.ndarray:
        """
        Apply subject-invariant normalization to remove subject identity signals.

        1. Temporal energy norm: X[i] /= rms(X[i]) per sample
        2. Global per-channel mean/std normalization
        3. Per-subject mean subtraction
        """
        if verbose:
            print("  [Preprocess] Subject-invariant normalization...")

        X = X.copy()

        # Step 1: Temporal energy normalization per sample
        # X shape: (N, C, T)
        rms = np.sqrt(np.mean(X**2, axis=(1, 2), keepdims=True)) + 1e-8
        X   = X / rms

        # Step 2: Global per-channel normalization (across all subjects)
        # Mean and std per channel across all samples
        global_mu  = X.mean(axis=(0, 2), keepdims=True)  # (1, C, 1)
        global_std = X.std(axis=(0, 2), keepdims=True) + 1e-8
        X = (X - global_mu) / global_std

        # Step 3: Per-subject mean subtraction
        # Removes any subject-specific DC offset
        unique_subjs = np.unique(subjects)
        for sid in unique_subjs:
            mask            = subjects == sid
            subj_mean       = X[mask].mean(axis=0, keepdims=True)   # (1, C, T)
            X[mask]        -= subj_mean

        if verbose:
            print(f"  [Preprocess] After normalization: "
                  f"global mean={X.mean():.4f}, std={X.std():.4f}")

        return X.astype(np.float32)
