"""
PhysioNet EEGMMIDB Loader
==========================
Loads the EEG Motor Movement/Imagery Dataset using MNE-Python.

Dataset: https://physionet.org/content/eegmmidb/1.0.0/
  - 109 subjects
  - 64-channel EEG at 160 Hz
  - 14 runs per subject

Run → Semantic Category Mapping:
  runs [3,7,11]  → left fist imagery  → navigation
  runs [4,8,12]  → right fist imagery → action
  runs [5,9,13]  → both fists imagery → communication
  runs [6,10,14] → both feet imagery  → selection
  run  [1]       → eyes open baseline → idle

Falls back to SyntheticEEGGenerator if MNE or internet is unavailable.
"""

import numpy as np
from typing import Optional

from tsta_project.data.preprocess import EEGDataset
from tsta_project.data.synthetic.profiles import CATEGORIES


# Run number → semantic category mapping
RUN_TO_CATEGORY = {
    **{r: 1 for r in [3, 7, 11]},    # left fist  → navigation
    **{r: 2 for r in [4, 8, 12]},    # right fist → action
    **{r: 0 for r in [5, 9, 13]},    # both hands → communication
    **{r: 3 for r in [6, 10, 14]},   # feet       → selection
    **{r: 4 for r in [1]},           # baseline   → idle
}


def load_physionet(n_subjects: int = 5,
                   epoch_len:  float = 2.0) -> Optional[EEGDataset]:
    """
    Download and load PhysioNet EEGMMIDB.

    Args:
        n_subjects:  Number of subjects to load (1..109).
        epoch_len:   Epoch duration in seconds.

    Returns:
        EEGDataset on success, None on failure (triggers fallback in caller).
    """
    try:
        import mne
        mne.set_log_level("ERROR")
    except ImportError:
        print("  [PhysioNet] MNE not installed. Using synthetic fallback.")
        return None

    all_X, all_y, all_subj = [], [], []
    sfreq      = None
    ch_names   = None

    try:
        for subj in range(1, n_subjects + 1):
            for run, cat in RUN_TO_CATEGORY.items():
                try:
                    paths = mne.datasets.eegbci.load_data(
                        subj, [run], verbose=False
                    )
                    raw = mne.io.read_raw_edf(
                        paths[0], preload=True, verbose=False
                    )
                    mne.datasets.eegbci.standardize(raw)
                    raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)

                    if sfreq is None:
                        sfreq    = raw.info["sfreq"]
                        ch_names = raw.ch_names[:]

                    n_samp = int(epoch_len * raw.info["sfreq"])

                    if run == 1:
                        # Baseline: cut continuous recording into fixed-length epochs
                        data = raw.get_data()
                        for start in range(0, data.shape[1] - n_samp, n_samp):
                            ep = data[:, start:start + n_samp].astype(np.float32)
                            all_X.append(ep)
                            all_y.append(4)        # idle
                            all_subj.append(subj)
                    else:
                        events, _ = mne.events_from_annotations(raw, verbose=False)
                        if len(events) == 0:
                            continue
                        epochs = mne.Epochs(
                            raw, events,
                            tmin=0, tmax=epoch_len,
                            baseline=None,
                            preload=True,
                            verbose=False,
                        )
                        data = epochs.get_data().astype(np.float32)
                        for ep in data:
                            all_X.append(ep)
                            all_y.append(cat)
                            all_subj.append(subj)

                except Exception as e:
                    # Skip a single run if it fails; continue with others
                    continue

        if len(all_X) == 0:
            raise RuntimeError("No data loaded from PhysioNet.")

        X = np.array(all_X, dtype=np.float32)
        y = np.array(all_y, dtype=np.int64)
        s = np.array(all_subj, dtype=np.int64)

        print(f"  [PhysioNet] Loaded {len(y)} epochs | "
              f"{n_subjects} subjects | {X.shape[1]} channels | "
              f"{X.shape[2]} samples @ {sfreq}Hz")

        return EEGDataset(
            X=X,
            y=y,
            subjects=s,
            sfreq=float(sfreq),
            ch_names=ch_names or [f"EEG{i}" for i in range(X.shape[1])],
            categories=CATEGORIES,
            n_subjects=n_subjects,
            source="physionet",
        )

    except Exception as e:
        print(f"  [PhysioNet] Failed ({e}). Using synthetic fallback.")
        return None


def acquire_dataset(source: str = "auto",
                    n_subjects: int = 5,
                    n_per_class: int = 48) -> EEGDataset:
    """
    Load data from PhysioNet or fall back to synthetic.

    Args:
        source:       'physionet' | 'synthetic' | 'auto'
        n_subjects:   Number of subjects.
        n_per_class:  Trials per class (used only for synthetic).

    Returns:
        EEGDataset ready for preprocessing.
    """
    from tsta_project.data.synthetic.generator import SyntheticEEGGenerator

    if source == "synthetic":
        gen = SyntheticEEGGenerator(n_subjects=n_subjects, n_per_class=n_per_class)
        return gen.get_dataset()

    if source == "physionet":
        ds = load_physionet(n_subjects=n_subjects)
        if ds is None:
            raise RuntimeError("PhysioNet loading failed and fallback is disabled.")
        return ds

    # auto: try real, fall back to synthetic
    ds = load_physionet(n_subjects=n_subjects)
    if ds is None:
        print("  [Data] Falling back to synthetic dataset.")
        gen = SyntheticEEGGenerator(n_subjects=n_subjects, n_per_class=n_per_class)
        ds  = gen.get_dataset()
    return ds
