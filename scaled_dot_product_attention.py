import torch
import torch.nn as nn
import torch.nn.functional as F

# k, q, v
# attention of k, q, v
# softmax(qkt/root(dk))v


class ScaledDotProductAttnetion(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, K, Q, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask, -torch.inf)

        attn = F.softmax(scores, dim=-1)

        attn_scores = self.dropout(attn)

        output = torch.matmul(attn, V)

        return output, attn_scores

