
import torch, torch.nn as nn
class EEGNet(nn.Module):
    def __init__(self, n_classes, n_channels=64, n_timepoints=160, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,8,(1,64),padding=(0,32),bias=False), nn.BatchNorm2d(8))
        self.dw1 = nn.Sequential(nn.Conv2d(8,16,(n_channels,1),groups=8,bias=False), nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1,4)), nn.Dropout(dropout))
        self.sep = nn.Sequential(nn.Conv2d(16,16,(1,16),padding=(0,8),groups=16,bias=False), nn.Conv2d(16,16,1,bias=False), nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1,8)), nn.Dropout(dropout))
        self.fc = nn.Linear(16*(n_timepoints//32), n_classes)
    def forward(self, x):
        if x.dim()==3: x=x.unsqueeze(1)
        x=self.conv1(x); x=self.dw1(x); x=self.sep(x)
        return self.fc(x.view(x.size(0),-1))
