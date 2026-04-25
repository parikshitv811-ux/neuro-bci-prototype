
import torch, torch.nn.functional as F, numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

def compute_cdas(embeddings, labels, prototypes=None):
    embeddings = F.normalize(embeddings, p=2, dim=1)
    if prototypes is None:
        n_classes = int(labels.max().item()) + 1
        prototypes = torch.stack([embeddings[labels == c].mean(dim=0) if (labels == c).sum() > 0 else torch.zeros(embeddings.shape[1], device=embeddings.device) for c in range(n_classes)])
        prototypes = F.normalize(prototypes, p=2, dim=1)
    return np.mean([torch.dot(embeddings[i], prototypes[int(labels[i].item())]).item() for i in range(len(embeddings))])

def evaluate_model(model, dataloader, device='cpu'):
    model.eval(); all_preds, all_labels, all_emb = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['eeg'].to(device), batch['label'].to(device)
            subj = batch.get('subject', None)
            if subj is not None: subj = subj.to(device)
            logits, emb = model(x, subject_ids=subj)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(y.cpu().numpy()); all_emb.append(emb.cpu())
    all_emb = torch.cat(all_emb, dim=0)
    return {'accuracy': accuracy_score(all_labels, all_preds), 'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds), 'confusion_matrix': confusion_matrix(all_labels, all_preds), 'cdas': compute_cdas(all_emb, torch.tensor(all_labels), model.prototypes.cpu())}
