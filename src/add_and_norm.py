import torch.nn as nn

class AddAndNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mha, res):
        added = res + self.dropout(mha)
        out_norm = self.layer_norm(added)
        return out_norm

