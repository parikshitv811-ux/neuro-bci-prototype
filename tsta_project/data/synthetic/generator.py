"""
Synthetic EEG Generator
========================
Generates high-fidelity synthetic EEG signals that simulate real EEG
conditions with:
  - 64 channels (10-20 system layout)
  - 160 Hz sampling rate
  - Motor imagery / semantic intent patterns with temporal drift
  - Inter-subject variability (frequency offsets, noise level)
  - Pink (1/f) noise background
  - ERPs at physiologically correct latencies
  - Alpha ERD (event-related desynchronization)
  - Gaussian noise injection

Usage:
    gen = SyntheticEEGGenerator(n_subjects=5, n_per_class=48)
    ds  = gen.get_dataset()   # EEGDataset with X:(N,C,T), y:(N,), subjects:(N,)
"""

import numpy as np
from tsta_project.data.preprocess import EEGDataset
from tsta_project.data.synthetic.profiles import CATEGORIES, CATEGORY_PROFILES


class SyntheticEEGGenerator:
    """
    High-fidelity synthetic EEG generator matching PhysioNet EEGMMIDB structure.

    Simulated signal components per epoch:
      1. Pink (1/f) noise background
      2. Chirped oscillation (frequency drifts over time → semantic direction)
      3. ERP Gaussian bump at physiological latency
      4. Alpha ERD in central channels after cue onset
      5. Per-subject frequency offsets, amplitude scales, noise levels
    """

    SFREQ      = 160
    N_CHANNELS = 64
    EPOCH_LEN  = 2.0
    N_SAMPLES  = int(SFREQ * EPOCH_LEN)   # 320

    # 10-20 system region indices (approximate)
    FRONTAL   = list(range(0,  16))
    CENTRAL   = list(range(16, 32))
    PARIETAL  = list(range(32, 48))
    OCCIPITAL = list(range(48, 64))

    def __init__(self, n_subjects: int = 5, n_per_class: int = 48, seed: int = 0):
        self.n_subjects  = n_subjects
        self.n_per_class = n_per_class
        self.seed        = seed

    # ── Signal primitives ──────────────────────────────────────────────────────

    def _pink_noise(self, n: int, alpha: float = 1.0) -> np.ndarray:
        """Generate pink (1/f) noise of length n."""
        f = np.fft.rfftfreq(n)
        f[0] = 1e-10
        power    = f ** (-alpha / 2)
        spectrum = power * (np.random.randn(len(f)) + 1j * np.random.randn(len(f)))
        return np.real(np.fft.irfft(spectrum, n=n))

    def _subject_profile(self, subj_id: int) -> dict:
        """Each subject has unique frequency offsets and noise levels (inter-subject variability)."""
        rng = np.random.RandomState(subj_id * 31337)
        return {
            "freq_offset":  rng.uniform(-2.0, 2.0),
            "amp_scale":    rng.uniform(0.8,  1.4),
            "noise_level":  rng.uniform(0.20, 0.45),
            "alpha_peak":   rng.uniform(9.0,  11.5),   # individual alpha frequency
            "phase_shift":  rng.uniform(0.0,  2 * np.pi),
        }

    def _generate_epoch(self,
                        cat_id: int,
                        subj_id: int,
                        trial_id: int,
                        rng: np.random.RandomState) -> np.ndarray:
        """Generate one (C, T) epoch for the given category and subject."""
        profile = self._subject_profile(subj_id)
        prof    = CATEGORY_PROFILES[cat_id]
        t       = np.linspace(0, self.EPOCH_LEN, self.N_SAMPLES)
        epoch   = np.zeros((self.N_CHANNELS, self.N_SAMPLES), dtype=np.float32)

        f0, f1     = prof["band"]
        f0 += profile["freq_offset"]
        f1 += profile["freq_offset"]
        drift_lo, drift_hi = prof["drift"]

        # Temporal frequency chirp — encodes the semantic direction
        freq_t = np.linspace(f0 + drift_lo, f1 + drift_hi, self.N_SAMPLES)

        # Region-specific amplitude weighting
        lat = prof["laterality"]
        region_weights = {
            "left":      {r: 1.0 for r in self.CENTRAL[:8]},
            "right":     {r: 1.0 for r in self.CENTRAL[8:]},
            "bilateral": {r: 0.8 for r in self.CENTRAL},
            "parietal":  {r: 1.2 for r in self.PARIETAL},
            "none":      {},
        }[lat]

        for ch in range(self.N_CHANNELS):
            # 1/f background noise
            bg = self._pink_noise(self.N_SAMPLES) * 0.3 * profile["amp_scale"]

            # Chirped oscillation (temporal direction signal)
            phase_t = 2 * np.pi * np.cumsum(freq_t) / self.SFREQ
            ph_off  = rng.uniform(0, 2 * np.pi) + profile["phase_shift"]
            amp_ch  = profile["amp_scale"] * region_weights.get(ch, 0.4)
            osc     = amp_ch * np.sin(phase_t + ph_off)

            # ERP component at physiological latency
            if prof["erp_ms"] is not None and ch in (self.PARIETAL + self.CENTRAL):
                erp_t     = prof["erp_ms"] / 1000.0
                erp_idx   = int(erp_t * self.SFREQ)
                erp_width = int(0.08 * self.SFREQ)    # 80ms Gaussian width
                if erp_idx < self.N_SAMPLES:
                    g = np.exp(
                        -0.5 * ((np.arange(self.N_SAMPLES) - erp_idx) / erp_width) ** 2
                    )
                    osc += g * rng.uniform(0.5, 1.5)

            # Alpha ERD in central channels after cue onset (cognitive suppression)
            if cat_id != 4 and ch in self.CENTRAL:
                alpha_ph = (
                    2 * np.pi * profile["alpha_peak"] * t
                    + rng.uniform(0, 2 * np.pi)
                )
                erd_mask = np.where(t > 0.5,
                                    np.exp(-(t - 0.5) * 3.0) * (-0.6), 0)
                osc += erd_mask * np.sin(alpha_ph)

            # Gaussian noise
            noise = rng.randn(self.N_SAMPLES) * profile["noise_level"]

            epoch[ch] = bg + osc + noise

        return epoch

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_dataset(self) -> EEGDataset:
        """
        Generate the full dataset.

        Returns:
            EEGDataset with:
              X:        (N, C, T) float32
              y:        (N,)      int labels
              subjects: (N,)      int subject IDs
        """
        rng = np.random.RandomState(self.seed)
        all_X, all_y, all_subj = [], [], []

        total = self.n_subjects * len(CATEGORIES) * self.n_per_class
        print(f"  [Synthetic] Generating {self.n_subjects} subjects × "
              f"{len(CATEGORIES)} classes × {self.n_per_class} trials "
              f"= {total} epochs...")

        for subj in range(1, self.n_subjects + 1):
            for cat in range(len(CATEGORIES)):
                for trial in range(self.n_per_class):
                    ep = self._generate_epoch(cat, subj, trial, rng)
                    all_X.append(ep)
                    all_y.append(cat)
                    all_subj.append(subj)

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)
        s = np.array(all_subj, dtype=np.int64)

        ch_names = [f"EEG{i:03d}" for i in range(self.N_CHANNELS)]
        print(f"  [Synthetic] Dataset shape: {X.shape}  "
              f"sfreq={self.SFREQ}Hz  subjects={self.n_subjects}")

        return EEGDataset(
            X=X,
            y=y,
            subjects=s,
            sfreq=float(self.SFREQ),
            ch_names=ch_names,
            categories=CATEGORIES,
            n_subjects=self.n_subjects,
            source="synthetic",
        )
