
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np, os, json, sys
sys.path.insert(0, '/content/neuro-bci-prototype')
from tsta_project.models_v2 import TSTAEncoder
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

# -- Helper Functions --
class EEGDataset(Dataset):
    def __init__(self, X, y, subject_ids):
        self.X = torch.FloatTensor(X)
        unique_labels = sorted(np.unique(y))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        self.y = torch.LongTensor([label_map[label] for label in y])
        self.subj = torch.LongTensor(subject_ids)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {'eeg': self.X[idx], 'label': self.y[idx], 'subject': self.subj[idx]}

def train_loso():
    dataset = PhysionetMI(); paradigm = MotorImagery(fmin=4, fmax=40, tmin=0, tmax=4)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3]) # Start with 3 subjects for speed
    subject_ids = meta['subject'].astype(int).values
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logo = LeaveOneGroupOut()
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, labels, subject_ids)):
        print(f"\nFOLD {fold+1} | Test Subject: {np.unique(subject_ids[test_idx])}")
        train_set = EEGDataset(X[train_idx], labels[train_idx], subject_ids[train_idx])
        test_set = EEGDataset(X[test_idx], labels[test_idx], subject_ids[test_idx])
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=32)

        model = TSTAEncoder(n_channels=X.shape[1], n_timepoints=X.shape[2], n_classes=len(np.unique(labels)), n_subjects=110).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
        for epoch in range(10):
            model.train(); total_loss = 0
            for batch in train_loader:
                x, y, s = batch['eeg'].to(device), batch['label'].to(device), batch['subject'].to(device)
                optimizer.zero_grad(); logits, _ = model(x, subject_ids=s)
                loss = F.cross_entropy(logits, y)
                loss.backward(); optimizer.step(); total_loss += loss.item()
            
            if epoch % 5 == 0:
                model.eval(); preds = []
                with torch.no_grad():
                    for b in test_loader: 
                        l, _ = model(b['eeg'].to(device), subject_ids=b['subject'].to(device))
                        preds.extend(l.argmax(1).cpu().numpy())
                acc = accuracy_score(test_set.y.numpy(), preds)
                print(f"  Ep {epoch}: Loss {total_loss/len(train_loader):.4f} | Test Acc {acc:.3f}")

if __name__ == '__main__':
    print("🚀 Starting TSTA LOSO Validation...")
    train_loso()
