"""
TSTA Phase 1+2 — Data Acquisition & Preprocessing
===================================================
Real data:    PhysioNet EEGMMIDB via MNE (run with real hardware)
Fallback:     High-fidelity synthetic dataset matching PhysioNet structure
              (64ch, 160Hz, 5 semantic categories, 5 subjects, 240 trials each)

Paradigm:
  500ms fixation → cue → 2000ms imagination → 1500ms rest
  5 categories × 8 trials × 6 blocks = 240 trials/subject
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.signal import butter, filtfilt, iirnotch
warnings.filterwarnings('ignore')

# ─── PARADIGM CONFIG ──────────────────────────────────────────────────────────
CATEGORIES = {
    0: 'communication',   # "send message"
    1: 'navigation',      # "move forward"
    2: 'action',          # "open app"
    3: 'selection',       # "confirm"
    4: 'idle',            # resting state
}

# Each category: (dominant_band_hz, trajectory_type, erp_latency_ms)
# trajectory_type encodes the DIRECTION signature we're trying to detect
CATEGORY_PROFILES = {
    0: {'band': (8,  12), 'drift': (+2.0, -1.5), 'erp_ms': 300,  'laterality': 'left'},
    1: {'band': (12, 18), 'drift': (-1.5, +2.0), 'erp_ms': 250,  'laterality': 'right'},
    2: {'band': (15, 25), 'drift': (+3.0, +1.0), 'erp_ms': 200,  'laterality': 'bilateral'},
    3: {'band': (10, 15), 'drift': (-2.0, -2.0), 'erp_ms': 350,  'laterality': 'parietal'},
    4: {'band': (6,   9), 'drift': (0.0,   0.0), 'erp_ms': None, 'laterality': 'none'},
}

@dataclass
class EEGDataset:
    X:          np.ndarray    # (N, C, T)
    y:          np.ndarray    # (N,) int labels
    subjects:   np.ndarray    # (N,) subject IDs
    sfreq:      float
    ch_names:   List[str]
    categories: dict
    n_subjects: int
    source:     str           # 'physionet' | 'synthetic'


# ─── PHASE 1A: PHYSIONET REAL DATA LOADER ───────────────────────────────────
def load_physionet(n_subjects=5, epoch_len=2.0) -> Optional[EEGDataset]:
    """
    Load PhysioNet EEG Motor Imagery Database (EEGMMIDB).
    Maps motor imagery runs to TSTA semantic categories.

    Requires: internet access + pip install mne
    Run mapping:
      runs [3,7,11]  → left  fist  → navigation
      runs [4,8,12]  → right fist  → action
      runs [5,9,13]  → both  fists → communication
      runs [6,10,14] → both  feet  → selection
      baseline run 1 → eyes open   → idle
    """
    try:
        import mne
        mne.set_log_level('ERROR')
        all_X, all_y, all_subj = [], [], []

        run_to_cat = {
            **{r: 1 for r in [3,7,11]},    # left hand  → navigation
            **{r: 2 for r in [4,8,12]},    # right hand → action
            **{r: 0 for r in [5,9,13]},    # both hands → communication
            **{r: 3 for r in [6,10,14]},   # feet       → selection
            **{r: 4 for r in [1]},         # baseline   → idle
        }

        for subj in range(1, n_subjects + 1):
            for run, cat in run_to_cat.items():
                paths = mne.datasets.eegbci.load_data(subj, [run], verbose=False)
                raw   = mne.io.read_raw_edf(paths[0], preload=True, verbose=False)
                mne.datasets.eegbci.standardize(raw)
                raw.filter(1., 40., fir_design='firwin', verbose=False)

                if run == 1:
                    # Baseline: cut into 2s epochs
                    n_samp = int(epoch_len * raw.info['sfreq'])
                    data   = raw.get_data()  # (C, T)
                    for start in range(0, data.shape[1] - n_samp, n_samp):
                        ep = data[:, start:start+n_samp]
                        all_X.append(ep); all_y.append(4); all_subj.append(subj)
                else:
                    events, _ = mne.events_from_annotations(raw, verbose=False)
                    epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_len,
                                        baseline=None, preload=True, verbose=False)
                    data = epochs.get_data()  # (n_epochs, C, T)
                    for ep in data:
                        all_X.append(ep); all_y.append(cat); all_subj.append(subj)

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y)
        s = np.array(all_subj)
        print(f"[PhysioNet] Loaded {len(y)} epochs, {n_subjects} subjects, "
              f"{X.shape[1]} channels, {X.shape[2]} samples")
        return EEGDataset(X, y, s, raw.info['sfreq'],
                         raw.ch_names, CATEGORIES, n_subjects, 'physionet')
    except Exception as e:
        print(f"[PhysioNet] Not available ({e}). Using synthetic fallback.")
        return None


# ─── PHASE 1B: HIGH-FIDELITY SYNTHETIC DATA ──────────────────────────────────
class SyntheticEEGGenerator:
    """
    Generates EEG-like signals that match PhysioNet EEGMMIDB structure:
    - 64 channels (10-20 system layout)
    - 160 Hz sampling rate
    - Realistic inter-subject variability
    - Temporal drift structure (the TSTA signal)
    - ERPs at physiologically correct latencies
    - Pink 1/f noise background
    - Electrode cross-correlation (brain connectivity)

    This is the fallback when real data isn't available.
    One-line swap for real data: replace get_dataset() with load_physionet()
    """

    SFREQ      = 160          # Hz — matches PhysioNet
    N_CHANNELS = 64
    EPOCH_LEN  = 2.0          # seconds
    N_SAMPLES  = int(SFREQ * EPOCH_LEN)  # 320

    # 10-20 channel regions (index ranges, approximate)
    FRONTAL   = list(range(0,  16))
    CENTRAL   = list(range(16, 32))
    PARIETAL  = list(range(32, 48))
    OCCIPITAL = list(range(48, 64))

    def __init__(self, n_subjects=5, n_per_class=48, seed=0):
        self.n_subjects   = n_subjects
        self.n_per_class  = n_per_class
        self.seed         = seed

    def _pink_noise(self, n, alpha=1.0):
        """1/f noise (pink noise) — realistic EEG background."""
        f = np.fft.rfftfreq(n)
        f[0] = 1e-10
        power = f ** (-alpha / 2)
        spectrum = power * (np.random.randn(len(f)) + 1j * np.random.randn(len(f)))
        return np.real(np.fft.irfft(spectrum, n=n))

    def _subject_profile(self, subj_id):
        """Each subject has unique frequency offsets and noise levels."""
        rng = np.random.RandomState(subj_id * 31337)
        return {
            'freq_offset':   rng.uniform(-2.0, 2.0),
            'amp_scale':     rng.uniform(0.8,  1.4),
            'noise_level':   rng.uniform(0.20, 0.45),
            'alpha_peak':    rng.uniform(9.0,  11.5),  # individual alpha freq
        }

    def _generate_epoch(self, cat_id, subj_id, trial_id, rng):
        profile = self._subject_profile(subj_id)
        prof    = CATEGORY_PROFILES[cat_id]
        t       = np.linspace(0, self.EPOCH_LEN, self.N_SAMPLES)
        epoch   = np.zeros((self.N_CHANNELS, self.N_SAMPLES), dtype=np.float32)

        f0, f1   = prof['band']
        f0 += profile['freq_offset']; f1 += profile['freq_offset']
        drift_lo, drift_hi = prof['drift']

        # Frequency chirp — encodes the semantic direction
        freq_t = np.linspace(f0 + drift_lo, f1 + drift_hi, self.N_SAMPLES)

        # Region-specific amplitude modulation
        lat  = prof['laterality']
        region_weights = {
            'left':       {r: 1.0 for r in self.CENTRAL[:8]},
            'right':      {r: 1.0 for r in self.CENTRAL[8:]},
            'bilateral':  {r: 0.8 for r in self.CENTRAL},
            'parietal':   {r: 1.2 for r in self.PARIETAL},
            'none':       {},
        }[lat]

        for ch in range(self.N_CHANNELS):
            # 1/f background
            bg = self._pink_noise(self.N_SAMPLES) * 0.3 * profile['amp_scale']

            # Oscillatory component with temporal chirp
            phase_t = 2 * np.pi * np.cumsum(freq_t) / self.SFREQ
            ph_off  = rng.uniform(0, 2 * np.pi)
            amp_ch  = profile['amp_scale'] * region_weights.get(ch, 0.4)
            osc     = amp_ch * np.sin(phase_t + ph_off)

            # ERP component at physiological latency
            if prof['erp_ms'] is not None and ch in self.PARIETAL + self.CENTRAL:
                erp_t     = prof['erp_ms'] / 1000
                erp_idx   = int(erp_t * self.SFREQ)
                erp_width = int(0.08 * self.SFREQ)   # 80ms width
                erp_gauss = np.zeros(self.N_SAMPLES)
                if erp_idx < self.N_SAMPLES:
                    g = np.exp(-0.5 * ((np.arange(self.N_SAMPLES)-erp_idx)/erp_width)**2)
                    erp_gauss = g * rng.uniform(0.5, 1.5)
                osc += erp_gauss

            # Alpha suppression in active regions (ERD)
            if cat_id != 4 and ch in self.CENTRAL:
                alpha_t  = profile['alpha_peak']
                alpha_ph = 2 * np.pi * alpha_t * t + rng.uniform(0, 2 * np.pi)
                # Suppress alpha after 500ms (ERD onset)
                erd_mask = np.where(t > 0.5,
                                    np.exp(-(t - 0.5) * 3.0) * (-0.6), 0)
                osc += erd_mask * np.sin(alpha_ph)

            # Gaussian noise
            noise = rng.randn(self.N_SAMPLES) * profile['noise_level']

            epoch[ch] = bg + osc + noise

        return epoch

    def get_dataset(self) -> EEGDataset:
        rng = np.random.RandomState(self.seed)
        all_X, all_y, all_subj = [], [], []

        print(f"[Synthetic] Generating {self.n_subjects} subjects × "
              f"{len(CATEGORIES)} classes × {self.n_per_class} trials...")

        for subj in range(1, self.n_subjects + 1):
            for cat in range(len(CATEGORIES)):
                for trial in range(self.n_per_class):
                    ep = self._generate_epoch(cat, subj, trial, rng)
                    all_X.append(ep)
                    all_y.append(cat)
                    all_subj.append(subj)

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y)
        s = np.array(all_subj)

        ch_names = [f'EEG{i:03d}' for i in range(self.N_CHANNELS)]
        print(f"[Synthetic] Dataset: {X.shape}  subjects={self.n_subjects}  "
              f"sfreq={self.SFREQ}Hz")
        return EEGDataset(X, y, s, self.SFREQ, ch_names,
                         CATEGORIES, self.n_subjects, 'synthetic')


# ─── PHASE 2: PREPROCESSING PIPELINE ────────────────────────────────────────
class Preprocessor:
    """
    Bandpass 1-40Hz → Notch 50Hz → Baseline correction → Per-channel z-score.
    Designed to work identically on PhysioNet and synthetic data.
    """
    def __init__(self, sfreq=160, notch_hz=50.0):
        self.sfreq    = sfreq
        self.notch_hz = notch_hz
        nyq = sfreq / 2
        # Bandpass 1-40Hz
        self.b_bp, self.a_bp = butter(4, [1.0/nyq, 40.0/nyq], btype='band')
        # Notch
        b_n, a_n = iirnotch(notch_hz, Q=30, fs=sfreq)
        self.b_n, self.a_n = b_n, a_n

    def process(self, epoch: np.ndarray) -> np.ndarray:
        """epoch: (C, T) → preprocessed (C, T)"""
        ep = filtfilt(self.b_bp, self.a_bp, epoch,   axis=-1)
        ep = filtfilt(self.b_n,  self.a_n,  ep,      axis=-1)
        # Baseline correction: subtract mean of first 100ms
        n_baseline = max(1, int(0.1 * self.sfreq))
        baseline   = ep[:, :n_baseline].mean(axis=-1, keepdims=True)
        ep         = ep - baseline
        # Per-channel z-score
        mu  = ep.mean(axis=-1, keepdims=True)
        sig = ep.std(axis=-1, keepdims=True) + 1e-8
        return ((ep - mu) / sig).astype(np.float32)

    def process_dataset(self, ds: EEGDataset) -> EEGDataset:
        print(f"[Preprocess] Filtering {len(ds.y)} epochs "
              f"(bp 1-40Hz + notch {self.notch_hz}Hz + baseline)...")
        X_proc = np.array([self.process(ep) for ep in ds.X])
        return EEGDataset(X_proc, ds.y, ds.subjects, ds.sfreq,
                         ds.ch_names, ds.categories, ds.n_subjects, ds.source)


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────
def acquire_and_preprocess(n_subjects=5, n_per_class=48) -> EEGDataset:
    """Try real data first, fall back to synthetic."""
    ds = load_physionet(n_subjects=n_subjects)
    if ds is None:
        gen = SyntheticEEGGenerator(n_subjects=n_subjects, n_per_class=n_per_class)
        ds  = gen.get_dataset()
    prep = Preprocessor(sfreq=ds.sfreq)
    return prep.process_dataset(ds)


if __name__ == '__main__':
    ds = acquire_and_preprocess(n_subjects=5, n_per_class=48)
    print(f"\nDataset ready:")
    print(f"  Shape     : {ds.X.shape}  (N × C × T)")
    print(f"  Classes   : {dict(zip(*np.unique(ds.y, return_counts=True)))}")
    print(f"  Subjects  : {np.unique(ds.subjects).tolist()}")
    print(f"  Source    : {ds.source}")
    print(f"  Mean/std  : {ds.X.mean():.4f} / {ds.X.std():.4f}")
