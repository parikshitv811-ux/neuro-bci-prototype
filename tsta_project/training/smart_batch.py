"""
Smart Batching — Subject-Aware Batch Sampler
=============================================
Each batch MUST contain the same class from multiple subjects.
At least `min_subjects_per_class` subjects per class per batch.

Without this, cross-subject direction alignment fails because the model
never sees same-class data from different subjects in the same update step.

Usage:
    sampler = SubjectAwareBatchSampler(
        y, subjects, batch_size=32, min_subjects_per_class=3
    )
    loader = DataLoader(dataset, batch_sampler=sampler)
"""

import numpy as np
from torch.utils.data import Sampler


class SubjectAwareBatchSampler(Sampler):
    """
    Builds batches where each semantic class is represented by
    samples from at least `min_subjects_per_class` distinct subjects.

    Algorithm:
      1. Build index map: {class: {subject: [sample_idx, ...]}}
      2. For each batch:
         a. Randomly pick a pivot class
         b. Pick `n_per_class_per_subj` samples from each subject × class
         c. Fill remaining slots randomly from all classes
    """

    def __init__(self,
                 y:                      np.ndarray,
                 subjects:               np.ndarray,
                 batch_size:             int   = 32,
                 min_subjects_per_class: int   = 3,
                 n_per_class_per_subj:   int   = 2,
                 seed:                   int   = 42,
                 drop_last:              bool  = True):
        self.batch_size             = batch_size
        self.min_subjects_per_class = min_subjects_per_class
        self.n_per_class_per_subj   = n_per_class_per_subj
        self.drop_last              = drop_last
        self._rng                   = np.random.RandomState(seed)

        # Build index map: class → {subject → [idx]}
        classes  = np.unique(y)
        subs     = np.unique(subjects)
        self._idx_map = {
            c: {
                s: np.where((y == c) & (subjects == s))[0].tolist()
                for s in subs
                if ((y == c) & (subjects == s)).any()
            }
            for c in classes
        }
        self._all_idx = np.arange(len(y))
        self._n       = len(y)

    def _build_batches(self):
        batches = []
        n_batches = self._n // self.batch_size
        for _ in range(n_batches):
            batch = []
            # For each class, pick from multiple subjects
            for cls, subj_map in self._idx_map.items():
                subjs = list(subj_map.keys())
                if len(subjs) >= self.min_subjects_per_class:
                    chosen = self._rng.choice(
                        subjs,
                        size=min(self.min_subjects_per_class, len(subjs)),
                        replace=False,
                    )
                else:
                    chosen = subjs
                for s in chosen:
                    pool = subj_map[s]
                    if pool:
                        picks = self._rng.choice(
                            pool,
                            size=min(self.n_per_class_per_subj, len(pool)),
                            replace=len(pool) < self.n_per_class_per_subj,
                        )
                        batch.extend(picks.tolist())

            # Fill to batch_size with random samples
            while len(batch) < self.batch_size:
                batch.append(int(self._rng.choice(self._all_idx)))

            batches.append(batch[:self.batch_size])
        return batches

    def __iter__(self):
        batches = self._build_batches()
        self._rng.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self):
        return self._n // self.batch_size
