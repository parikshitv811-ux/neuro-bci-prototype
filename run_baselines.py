
import sys, os, torch, torch.optim as optim, torch.nn as nn, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from tsta_project.baselines import EEGNet
# Since we are using a consolidated file for validation, we'll implement a simple LOSO loader here
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score

class EEGDataset(Dataset):
    def __init__(self, X, y, subject_ids):
        self.X = torch.FloatTensor(X)
        unique_labels = sorted(np.unique(y))
        label_map = {label: i for i, label in enumerate(unique_labels)}
        self.y = torch.LongTensor([label_map[label] for label in y])
        self.subj = torch.LongTensor(subject_ids)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return {'eeg': self.X[idx], 'label': self.y[idx], 'subject': self.subj[idx]}

def run_baselines():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = PhysionetMI(); paradigm = MotorImagery(fmin=4, fmax=40, tmin=0, tmax=4)
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[1, 2, 3])
    subject_ids = meta['subject'].astype(int).values
    
    logo = LeaveOneGroupOut()
    results = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, labels, subject_ids)):
        train_set = EEGDataset(X[train_idx], labels[train_idx], subject_ids[train_idx])
        test_set = EEGDataset(X[test_idx], labels[test_idx], subject_ids[test_idx])
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=32)

        model = EEGNet(n_classes=len(np.unique(labels)), n_channels=X.shape[1], n_timepoints=X.shape[2]).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(10):
            model.train()
            for batch in train_loader:
                x, y = batch['eeg'].to(device), batch['label'].to(device)
                optimizer.zero_grad(); loss = criterion(model(x), y); loss.backward(); optimizer.step()
        
        model.eval(); preds = []
        with torch.no_grad():
            for b in test_loader: preds.extend(model(b['eeg'].to(device)).argmax(1).cpu().numpy())
        acc = accuracy_score(test_set.y.numpy(), preds)
        results.append(acc)
        print(f"Subject {np.unique(subject_ids[test_idx])}: Acc={acc:.3f}")

    print(f"Mean EEGNet Accuracy: {np.mean(results):.3f}")

if __name__ == '__main__':
    run_baselines()
