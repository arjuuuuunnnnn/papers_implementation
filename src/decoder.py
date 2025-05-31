import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .multi_head_attention import MultiHeadAttention
from .add_and_norm import AddAndNorm
from .feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.masked_attn = MultiHeadAttention(dropout, d_model, num_heads)
        self.add_and_norm1 = AddAndNorm(d_model, dropout)
        self.encoder_attn = MultiHeadAttention(dropout, d_model, num_heads)
        self.add_and_norm2 = AddAndNorm(d_model, dropout)
        self.feed_forward = FeedForward(d_model, dropout)
        self.add_and_norm3 = AddAndNorm(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask):
        masked_attn_op = self.masked_attn(k=x, q=x, v=x, mask=mask)
        add_and_norm1_op = self.add_and_norm1(mha=masked_attn_op, res=x)
        encoder_attn_op = self.encoder_attn(k=encoder_output, q=add_and_norm1_op, v=encoder_output, mask=None)
        add_and_norm2_op = self.add_and_norm2(mha=encoder_attn_op, res=add_and_norm1_op)
        feed_forward_op = self.feed_forward(x=add_and_norm2_op)
        add_and_norm3_op = self.add_and_norm3(mha=feed_forward_op, res=add_and_norm2_op)
        return add_and_norm3_op

class Decoder(nn.Module):
    def __init__(self, num_layers=6, d_model=512, dropout=0.1, num_heads=8, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, mask=None):
        mask = self.causal_mask(x.size(1))
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, mask)

        return x

    def causal_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask
