
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery

class EEGDataset(Dataset):
    def __init__(self, X, y, subject_ids):
        self.X = torch.FloatTensor(X)
        # Convert string labels to integers
        if isinstance(y[0], (str, np.str_)):
            unique_labels = sorted(np.unique(y))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
        self.y = torch.LongTensor(y)
        self.subj = torch.LongTensor(subject_ids)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {'eeg': self.X[idx], 'label': self.y[idx], 'subject': self.subj[idx]}

def get_loso_loaders(dataset_name='PhysionetMI', subjects=None, fmin=4, fmax=40, tmin=0, tmax=4):
    dataset = PhysionetMI(); paradigm = MotorImagery(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    subject_ids = meta['subject'].astype(int).values
    print(f"Samples: {len(X)}, Subjects: {len(np.unique(subject_ids))}, Shape: {X.shape}")
    splits = []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, labels, groups=subject_ids):
        X_tr, y_tr, X_te, y_te = X[train_idx], labels[train_idx], X[test_idx], labels[test_idx]
        s_tr, s_te = subject_ids[train_idx], subject_ids[test_idx]
        unique_tr = np.unique(s_tr); s_map = {s: i for i, s in enumerate(unique_tr)}
        splits.append({'train': DataLoader(EEGDataset(X_tr, y_tr, np.array([s_map[s] for s in s_tr])), batch_size=32, shuffle=True), 'test': DataLoader(EEGDataset(X_te, y_te, np.array([s_map.get(s, 0) for s in s_te])), batch_size=32, shuffle=False), 'test_subject': int(np.unique(s_te)[0]), 'n_train_subjects': len(unique_tr)})
    print(f"LOSO splits: {len(splits)}")
    return splits
