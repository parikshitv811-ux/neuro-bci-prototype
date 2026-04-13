"""
BCI Level 2 — Hardware Integration & EEG Paradigms
====================================================
BrainFlow real-device connector + SSVEP and P300 paradigm implementations.
In simulation mode (no hardware), uses realistic synthetic signals.
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────
# HARDWARE ABSTRACTION LAYER
# ─────────────────────────────────────────────────────────────
class EEGSource:
    """Abstract base — swap simulation for real hardware by subclassing."""
    N_CHANNELS: int = 14
    SFREQ: int = 256

    def start(self): pass
    def stop(self): pass

    def get_epoch(self, n_samples: int) -> np.ndarray:
        """Return (n_channels, n_samples) float32 array."""
        raise NotImplementedError


class SimulatedEEGSource(EEGSource):
    """Drop-in simulation for development without hardware."""

    def __init__(self, sfreq=256, n_channels=14):
        self.SFREQ = sfreq
        self.N_CHANNELS = n_channels

    def get_epoch(self, n_samples: int) -> np.ndarray:
        t = np.linspace(0, n_samples / self.SFREQ, n_samples)
        data = np.zeros((self.N_CHANNELS, n_samples), dtype=np.float32)
        for ch in range(self.N_CHANNELS):
            freq = np.random.uniform(8, 13)
            data[ch] = np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2*np.pi))
            data[ch] += 0.3 * np.random.randn(n_samples)
        return data


class BrainFlowSource(EEGSource):
    """
    Real hardware via BrainFlow API.
    Supports: OpenBCI Cyton (board_id=0), Ganglion (board_id=1),
              Muse 2 (board_id=38), Synthetic (board_id=-1 for testing).

    Usage:
        src = BrainFlowSource(board_id=0, serial_port='/dev/ttyUSB0')
        src.start()
        epoch = src.get_epoch(512)   # 2s at 256Hz
        src.stop()
    """
    def __init__(self, board_id: int = -1, serial_port: str = '',
                 ip_address: str = '', ip_port: int = 0):
        self.board_id = board_id
        self.serial_port = serial_port
        self.ip_address = ip_address
        self.ip_port = ip_port
        self._board = None
        self._eeg_channels = None

        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
            from brainflow.data_filter import DataFilter, FilterTypes
            self._BoardShim = BoardShim
            self._BrainFlowInputParams = BrainFlowInputParams
            self._BoardIds = BoardIds
            self._DataFilter = DataFilter
            self._FilterTypes = FilterTypes
            self._available = True
            self.SFREQ = BoardShim.get_sampling_rate(board_id)
            self._eeg_channels = BoardShim.get_eeg_channels(board_id)
            self.N_CHANNELS = len(self._eeg_channels)
        except ImportError:
            print("[BrainFlowSource] brainflow not installed — use SimulatedEEGSource")
            self._available = False

    def start(self):
        if not self._available:
            return
        params = self._BrainFlowInputParams()
        if self.serial_port:
            params.serial_port = self.serial_port
        if self.ip_address:
            params.ip_address = self.ip_address
        if self.ip_port:
            params.ip_port = self.ip_port

        self._board = self._BoardShim(self.board_id, params)
        self._board.prepare_session()
        self._board.start_stream()
        time.sleep(2)  # allow buffer to fill
        print(f"[BrainFlowSource] Stream started — {self.N_CHANNELS}ch @ {self.SFREQ}Hz")

    def stop(self):
        if self._board:
            self._board.stop_stream()
            self._board.release_session()

    def get_epoch(self, n_samples: int) -> np.ndarray:
        if not self._board:
            raise RuntimeError("Board not started. Call start() first.")
        data = self._board.get_current_board_data(n_samples)
        return data[self._eeg_channels].astype(np.float32)


# ─────────────────────────────────────────────────────────────
# SSVEP PARADIGM
# ─────────────────────────────────────────────────────────────
@dataclass
class SSVEPConfig:
    """
    Steady-State Visual Evoked Potential configuration.

    User stares at a flickering stimulus at frequency F Hz.
    The occipital EEG shows a peak at exactly F Hz.
    Multiple stimuli at different freqs → multi-class selection.
    """
    frequencies: Tuple[float, ...] = (8.0, 10.0, 12.0, 15.0)
    labels: Tuple[str, ...] = ("activate", "select", "cancel", "idle")
    epoch_duration_s: float = 2.0
    occipital_channels: Tuple[int, ...] = (10, 11, 12, 13)  # O1,O2,Oz,POz
    harmonic_weights: Tuple[float, ...] = (1.0, 0.5)        # fundamental + 2nd harmonic


class SSVEPDetector:
    """
    Detects SSVEP responses using power spectral analysis at target frequencies.
    Returns the stimulus frequency with the highest SNR.

    Accuracy: 70-90% with real hardware, higher with longer epochs.
    """

    def __init__(self, config: SSVEPConfig, sfreq: int = 256):
        self.cfg = config
        self.sfreq = sfreq
        self.n_samples = int(config.epoch_duration_s * sfreq)

    def _compute_snr(self, psd: np.ndarray, freqs: np.ndarray, target_hz: float) -> float:
        """SNR at target frequency relative to surrounding noise floor."""
        snr = 0.0
        for i, w in enumerate(self.cfg.harmonic_weights):
            f_target = target_hz * (i + 1)
            f_noise_lo = f_target - 1.0
            f_noise_hi = f_target + 1.0
            # Signal power at target (±0.3 Hz bin)
            sig_mask = (freqs >= f_target - 0.3) & (freqs <= f_target + 0.3)
            noise_mask = ((freqs >= f_noise_lo) & (freqs < f_target - 0.3)) | \
                         ((freqs > f_target + 0.3) & (freqs <= f_noise_hi))
            if sig_mask.any() and noise_mask.any():
                snr += w * (psd[sig_mask].mean() / (psd[noise_mask].mean() + 1e-12))
        return snr

    def classify(self, epoch: np.ndarray) -> dict:
        """
        Args:
            epoch: (n_channels, n_samples)
        Returns:
            dict with 'label', 'frequency', 'confidence', 'snr_scores'
        """
        from scipy.signal import welch

        # Use occipital channels only
        occ = [ch for ch in self.cfg.occipital_channels if ch < epoch.shape[0]]
        if not occ:
            occ = list(range(epoch.shape[0]))
        signal = epoch[occ].mean(axis=0)

        freqs, psd = welch(signal, fs=self.sfreq, nperseg=min(256, self.n_samples))

        snr_scores = []
        for f in self.cfg.frequencies:
            snr_scores.append(self._compute_snr(psd, freqs, f))

        best_idx = int(np.argmax(snr_scores))
        total = sum(snr_scores) + 1e-9
        confidence = snr_scores[best_idx] / total

        return {
            "label": self.cfg.labels[best_idx],
            "frequency": self.cfg.frequencies[best_idx],
            "confidence": round(confidence, 4),
            "snr_scores": {
                self.cfg.labels[i]: round(s, 4)
                for i, s in enumerate(snr_scores)
            }
        }

    def simulate_response(self, target_freq: float, noise_level: float = 0.3) -> np.ndarray:
        """Generate synthetic SSVEP epoch for a given stimulus frequency."""
        n_samples = self.n_samples
        t = np.linspace(0, self.cfg.epoch_duration_s, n_samples)
        data = np.zeros((14, n_samples), dtype=np.float32)
        for ch in range(14):
            phase = np.random.uniform(0, 2 * np.pi)
            # Strong SSVEP in occipital channels, weak elsewhere
            amp = 2.0 if ch in self.cfg.occipital_channels else 0.3
            data[ch] = amp * np.sin(2 * np.pi * target_freq * t + phase)
            data[ch] += 0.5 * amp * np.sin(2 * np.pi * target_freq * 2 * t + phase)
            data[ch] += noise_level * np.random.randn(n_samples)
        return data


# ─────────────────────────────────────────────────────────────
# P300 PARADIGM
# ─────────────────────────────────────────────────────────────
@dataclass
class P300Config:
    """
    P300 Event-Related Potential configuration.

    User counts target stimuli in an oddball sequence.
    A P300 (positive deflection ~300ms post-stimulus) appears
    in parietal channels (Pz, P3, P4) for target stimuli only.
    """
    n_stimuli: int = 10
    stimulus_duration_ms: int = 100
    isi_ms: int = 300                  # inter-stimulus interval
    target_probability: float = 0.2   # ~20% targets (oddball)
    parietal_channels: Tuple[int, ...] = (6, 7, 8)  # Pz, P3, P4
    p300_window_ms: Tuple[int, int] = (250, 500)


class P300Detector:
    """
    Detects P300 responses by averaging ERP across stimulus repetitions.
    Classifies target vs non-target based on parietal amplitude in 250-500ms window.
    """

    def __init__(self, config: P300Config, sfreq: int = 256):
        self.cfg = config
        self.sfreq = sfreq

    def _p300_amplitude(self, epoch: np.ndarray) -> float:
        """Mean amplitude in P300 window (250-500ms) on parietal channels."""
        lo = int(self.cfg.p300_window_ms[0] / 1000 * self.sfreq)
        hi = int(self.cfg.p300_window_ms[1] / 1000 * self.sfreq)
        hi = min(hi, epoch.shape[1])
        parietal = [ch for ch in self.cfg.parietal_channels if ch < epoch.shape[0]]
        if not parietal:
            parietal = [0]
        return float(epoch[parietal, lo:hi].mean())

    def classify_single(self, epoch: np.ndarray) -> dict:
        """
        Classify one post-stimulus epoch (500ms, ~128 samples at 256Hz).
        """
        amp = self._p300_amplitude(epoch)
        # Threshold-based (replace with trained SVM/LDA for production)
        threshold = 0.3
        is_target = amp > threshold
        confidence = min(abs(amp) / (threshold * 2 + 1e-9), 1.0)
        return {
            "is_target": is_target,
            "amplitude_uv": round(amp, 4),
            "confidence": round(confidence, 4)
        }

    def run_speller_step(self, epochs: list, labels: list) -> dict:
        """
        Average ERP across multiple presentations (improves SNR).
        epochs: list of (n_ch, n_samples) arrays
        labels: list of 0/1 (non-target/target)
        """
        targets = [ep for ep, l in zip(epochs, labels) if l == 1]
        nontargets = [ep for ep, l in zip(epochs, labels) if l == 0]
        if not targets:
            return {"selected": None, "confidence": 0.0}

        avg_target = np.mean(targets, axis=0)
        avg_nontarget = np.mean(nontargets, axis=0) if nontargets else np.zeros_like(avg_target)

        target_amp = self._p300_amplitude(avg_target)
        nontarget_amp = self._p300_amplitude(avg_nontarget)
        discriminability = target_amp - nontarget_amp
        confidence = min(max(discriminability / 1.0, 0.0), 1.0)

        return {
            "selected": discriminability > 0.2,
            "target_amplitude": round(target_amp, 4),
            "nontarget_amplitude": round(nontarget_amp, 4),
            "discriminability": round(discriminability, 4),
            "confidence": round(confidence, 4)
        }

    def simulate_epochs(self, n_target: int = 2, n_nontarget: int = 8) -> Tuple[list, list]:
        """Simulate P300 and non-P300 epochs for testing."""
        n_samples = int(0.5 * self.sfreq)  # 500ms epoch
        epochs, labels = [], []

        for _ in range(n_target):
            ep = 0.1 * np.random.randn(14, n_samples).astype(np.float32)
            for ch in self.cfg.parietal_channels:
                if ch < 14:
                    t_p300 = int(0.3 * self.sfreq)
                    ep[ch, t_p300:t_p300 + 30] += np.random.uniform(0.8, 1.2)
            epochs.append(ep); labels.append(1)

        for _ in range(n_nontarget):
            ep = 0.15 * np.random.randn(14, n_samples).astype(np.float32)
            epochs.append(ep); labels.append(0)

        idx = np.random.permutation(len(labels))
        return [epochs[i] for i in idx], [labels[i] for i in idx]
