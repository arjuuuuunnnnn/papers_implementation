import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, 4*d_model)#expansion 512->2048
        self.w_2 = nn.Linear(4*d_model, d_model)#compression 2048->512
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        return x

