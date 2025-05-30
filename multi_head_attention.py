import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dropout=0.1, d_model=512, num_heads=8):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.w_k = nn.Linear(d_model, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)#batch_size, num_heads, seq_length, head_dim
        return x

    def forward(self, k, q, v, mask=None):
        k = self.w_k(k)
        q = self.w_q(q)
        v = self.w_v(v)

        k_h = self.split_heads(k)
        q_h = self.split_heads(q)
        v_h = self.split_heads(v)

        scores = (q_h @ k_h.transpose(-2, -1)) / (self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask, -torch.inf)

        attn_scores = F.softmax(scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        context = attn_scores @ v_h

        context = context.permute(0, 2, 1, 3)#swap back heads and seq_length
        context = context.contiguous().view(context.shape[0], context.shape[1], self.d_model)#concat heads

        output_context = self.w_o(context)

        return output_context

