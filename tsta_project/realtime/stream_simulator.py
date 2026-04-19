"""
EEG Stream Simulator
=====================
Simulates a real-time EEG data stream by replaying synthetic epochs
through a sliding window, mimicking BCI hardware output at 160 Hz.

Features:
  - Sliding window with configurable chunk size (0.5–1.0 s)
  - Overlap-add streaming (50% overlap)
  - Injects per-class EEG signal appropriate for the semantic intent
  - Optional noise injection per chunk (SNR-based)
  - Yields (chunk, label, timestamp_ms) tuples
"""

import time
import numpy as np
from tsta_project.data.synthetic.generator  import SyntheticEEGGenerator
from tsta_project.data.synthetic.profiles   import CATEGORY_PROFILES, CATEGORIES
from tsta_project.config import TSTAConfig


class EEGStreamSimulator:
    """
    Simulates a continuous 64-channel EEG stream at 160 Hz.

    Usage:
        sim = EEGStreamSimulator(cfg, chunk_s=0.5)
        for chunk, label, t_ms in sim.stream(duration_s=10.0):
            ...  # chunk: (C, chunk_len)
    """

    SFREQ      = 160
    N_CHANNELS = 64

    def __init__(self,
                 cfg:        TSTAConfig,
                 chunk_s:    float = 0.5,
                 overlap:    float = 0.5,
                 n_subjects: int   = 5,
                 seed:       int   = 99,
                 noise_sigma: float = 0.3):
        self.cfg         = cfg
        self.chunk_len   = int(chunk_s  * self.SFREQ)
        self.step_len    = int(self.chunk_len * (1 - overlap))
        self.n_subjects  = n_subjects
        self.seed        = seed
        self.noise_sigma = noise_sigma
        self._rng        = np.random.RandomState(seed)

        # Pre-generate a large pool of epochs
        gen     = SyntheticEEGGenerator(n_subjects=n_subjects,
                                        n_per_class=20, seed=seed)
        self._ds = gen.get_dataset()
        self._X  = self._ds.X      # (N, C, T)
        self._y  = self._ds.y      # (N,)

    # ── Private ───────────────────────────────────────────────────────────────

    def _pick_epoch(self, cls: int | None = None) -> tuple[np.ndarray, int]:
        """Pick a random epoch, optionally from a specific class."""
        if cls is not None:
            idx = np.where(self._y == cls)[0]
        else:
            idx = np.arange(len(self._y))
        pick = self._rng.choice(idx)
        return self._X[pick].copy(), int(self._y[pick])

    # ── Public API ────────────────────────────────────────────────────────────

    def stream(self,
               duration_s:    float = 15.0,
               intent_seq:    list  = None,
               realtime_speed: bool = False):
        """
        Generator yielding EEG chunks in sequence.

        Args:
            duration_s:     Total stream duration in seconds
            intent_seq:     Optional list of class IDs to cycle through
            realtime_speed: If True, sleep to simulate real-time rate

        Yields:
            chunk: (N_CH, chunk_len) float32
            label: int (semantic class 0–4)
            t_ms:  float (simulated time in ms)
        """
        n_steps    = int(duration_s * self.SFREQ / self.step_len)
        intent_idx = 0
        buffer     = np.zeros((self.N_CHANNELS, self.chunk_len), dtype=np.float32)
        t_ms       = 0.0
        step_ms    = self.step_len / self.SFREQ * 1000.0

        # Current epoch tracking
        cur_epoch, cur_label = self._pick_epoch()
        epoch_pos            = 0
        T_epoch              = cur_epoch.shape[-1]

        for _ in range(n_steps):
            # Rotate intent
            if intent_seq is not None:
                cur_label = intent_seq[intent_idx % len(intent_seq)]
                cur_epoch, _ = self._pick_epoch(cls=cur_label)
                epoch_pos    = 0
                intent_idx  += (1 if _ % (self.SFREQ // self.step_len * 2) == 0 else 0)

            # Fill chunk from epoch (with wrap)
            chunk = np.zeros((self.N_CHANNELS, self.chunk_len), dtype=np.float32)
            for j in range(self.chunk_len):
                pos          = (epoch_pos + j) % T_epoch
                chunk[:, j]  = cur_epoch[:, pos]

            epoch_pos = (epoch_pos + self.step_len) % T_epoch

            # Add light noise
            chunk += self._rng.randn(*chunk.shape).astype(np.float32) * self.noise_sigma

            if realtime_speed:
                time.sleep(self.step_len / self.SFREQ)

            yield chunk.copy(), cur_label, t_ms
            t_ms += step_ms

    def stream_json(self, duration_s: float = 15.0, intent_seq: list = None):
        """
        Yields JSON-serialisable dicts for SSE streaming.
        """
        for chunk, label, t_ms in self.stream(duration_s=duration_s,
                                               intent_seq=intent_seq):
            # Downsample to 8 channels for bandwidth (frontal + central)
            ch_sel   = [0, 8, 16, 20, 24, 28, 32, 48]
            mini     = chunk[ch_sel, :].tolist()
            yield {
                "t_ms":      round(t_ms, 1),
                "label":     label,
                "intent":    CATEGORIES[label],
                "eeg_mini":  mini,       # 8 channels × chunk_len
                "n_samples": self.chunk_len,
                "sfreq":     self.SFREQ,
            }
