
import torch, torch.nn as nn, torch.nn.functional as F

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07): super().__init__(); self.temperature = temperature
    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        similarity = similarity - similarity.max(dim=1, keepdim=True)[0].detach()
        mask = labels.unsqueeze(0) == labels.unsqueeze(1); mask.fill_diagonal_(False)
        exp_sim = torch.exp(similarity); pos_sim = (exp_sim * mask).sum(dim=1)
        all_sim = exp_sim.sum(dim=1) - exp_sim.diag(); valid = mask.sum(dim=1) > 0
        return (-torch.log((pos_sim + 1e-8) / (all_sim + 1e-8)))[valid].mean() if valid.sum() > 0 else torch.tensor(0.0, device=embeddings.device)

class CanonicalAlignmentLoss(nn.Module):
    def __init__(self, lambda_reg=1e-4): super().__init__(); self.lambda_reg = lambda_reg
    def forward(self, embeddings, subject_ids):
        unique = torch.unique(subject_ids)
        if len(unique) < 2: return torch.tensor(0.0, device=embeddings.device)
        losses = []
        for i, si in enumerate(unique):
            for sj in unique[i+1:]:
                mi, mj = subject_ids == si, subject_ids == sj
                if mi.sum() < 2 or mj.sum() < 2: continue
                Zi = embeddings[mi] - embeddings[mi].mean(dim=0); Zj = embeddings[mj] - embeddings[mj].mean(dim=0)
                Ci = (Zi.T @ Zi) / (Zi.shape[0]-1) + self.lambda_reg * torch.eye(Zi.shape[1], device=Zi.device)
                Cj = (Zj.T @ Zj) / (Zj.shape[0]-1) + self.lambda_reg * torch.eye(Zj.shape[1], device=Zj.device)
                losses.append(torch.norm(Ci - Cj, p='fro'))
        return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=embeddings.device)

class PrototypeOrthogonalityLoss(nn.Module):
    def forward(self, prototypes):
        n = F.normalize(prototypes, p=2, dim=1)
        return torch.norm(torch.matmul(n, n.T) - torch.eye(n.shape[0], device=n.device), p='fro')

class CompositeLoss(nn.Module):
    def __init__(self, lambda_ca=0.1, lambda_proto=0.01, temperature=0.07):
        super().__init__()
        self.info_nce = InfoNCELoss(temperature); self.ca_loss = CanonicalAlignmentLoss(); self.proto_loss = PrototypeOrthogonalityLoss()
        self.lambda_ca = lambda_ca; self.lambda_proto = lambda_proto
    def forward(self, embeddings, labels, subject_ids, prototypes):
        nce = self.info_nce(embeddings, labels); ca = self.ca_loss(embeddings, subject_ids); proto = self.proto_loss(prototypes)
        return {'total': nce + self.lambda_ca * ca + self.lambda_proto * proto, 'nce': nce, 'ca': ca, 'proto': proto}
