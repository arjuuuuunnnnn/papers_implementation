import torch
import torch.nn as nn

import add_and_norm
from positional_encoding import PositionalEncoding
from multi_head_attention import MultiHeadAttention
from add_and_norm import AddAndNorm
from feed_forward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(dropout, d_model, num_heads)
        self.add_and_norm1 = AddAndNorm(d_model, dropout)
        self.feed_forward = FeedForward(d_model, dropout)
        self.add_and_norm2 = AddAndNorm(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        self_attn_op = self.self_attn(k=x, q=x, v=x, mask=mask)
        add_and_norm1_op = self.add_and_norm1(mha=self_attn_op, res=x)
        feed_forward_op = self.feed_forward(x=add_and_norm)
        add_and_norm2_op = self.add_and_norm2(mha=feed_forward_op, res=add_and_norm1_op)
        return add_and_norm2_op


class Encoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, num_heads=8, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x
