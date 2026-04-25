
import torch
import torch.nn as nn
import torch.nn.functional as F

class SubjectAdapter(nn.Module):
    def __init__(self, n_subjects, embed_dim=16, n_channels=64, hidden_dim=32):
        super().__init__()
        self.subject_embed = nn.Embedding(n_subjects, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_channels), nn.Sigmoid()
        )
    def forward(self, x, subject_ids):
        # x: (B, C, T)
        gate = self.gate(self.subject_embed(subject_ids)).unsqueeze(-1)
        return x * gate

class TSTAEncoder(nn.Module):
    def __init__(self, n_channels=64, n_timepoints=160, n_classes=4, n_subjects=109,
                 n_filters=(40, 40), kernel_sizes=(25, 15), dropout=0.5, embed_dim=128):
        super().__init__()
        self.adapter = SubjectAdapter(n_subjects, 16, n_channels)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, n_filters[0], (1, kernel_sizes[0]), padding=(0, kernel_sizes[0]//2)),
            nn.BatchNorm2d(n_filters[0]))
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(n_filters[0], n_filters[0], (n_channels, 1), groups=n_filters[0]),
            nn.BatchNorm2d(n_filters[0]), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.separable_conv = nn.Sequential(
            nn.Conv2d(n_filters[0], n_filters[1], (1, kernel_sizes[1]), padding=(0, kernel_sizes[1]//2)),
            nn.BatchNorm2d(n_filters[1]), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))

        # Dynamically calculate feature dim accurately
        with torch.no_grad():
            sample = torch.zeros(1, 1, n_channels, n_timepoints)
            sample = self.temporal_conv(sample)
            sample = self.depthwise_conv(sample)
            sample = self.separable_conv(sample)
            self.feature_dim = sample.view(1, -1).shape[1]

        self.projector = nn.Sequential(
            nn.Flatten(), nn.Linear(self.feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim), nn.ELU(), nn.Linear(embed_dim, embed_dim))
        self.prototypes = nn.Parameter(torch.randn(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x, subject_ids=None, return_embedding=False):
        if x.dim() == 3: x = x.unsqueeze(1)
        if subject_ids is not None: 
            # Apply adapter to (B, C, T) before convolution
            x_sq = x.squeeze(1)
            x = self.adapter(x_sq, subject_ids).unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        z = F.normalize(self.projector(x), p=2, dim=1)
        return z if return_embedding else (torch.matmul(z, self.prototypes.T), z)
