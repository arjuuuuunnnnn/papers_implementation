import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super(ScaledDotProductAttention, self).__init__()

        self.d_head = d_head

    def forward(self, q, k, v, mask=None):
        # q, k, v, dims = batch_size, n_heads, seq_len, d_head
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) # (batch_size, n_heads, seq_len, seq_len)
        scaled_attention_weights = attention_weights / math.sqrt(self.d_head)

        if mask is not None:
            scaled_attention_weights = scaled_attention_weights.masked_fill(mask == 0, float('-inf')) # (batch_size, n_heads, seq_len, seq_len)

        scaled_attention_weights = nn.functional.softmax(scaled_attention_weights, dim=-1)
        
        scaled_attention_weights = self.attention_dropout(scaled_attention_weights)

        weighted_v = torch.matmul(scaled_attention_weights, v)

        return weighted_v

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads

        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads

        self.dot_product_attention_layer = ScaledDotProductAttention(self.d_head)

        self.W_0 = nn.Linear(d_model, d_model)

    def _split_into_heads(self, q, k, v):
        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_head)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_head)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.d_head)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def _concatenate_heads(self, attention_output):
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(attention_output.size(0), attention_output.size(1), -1)
        return attention_output

    def forward(self, q, k, v, mask=None):
        q, k, v = self._split_into_heads(q, k, v)
        attention_output = self.dot_product_attention_layer(q, k, v, mask)
        attention_output = self._concatenate_heads(attention_output)
        attention_output = self.W_0(attention_output)

        return attention_output

