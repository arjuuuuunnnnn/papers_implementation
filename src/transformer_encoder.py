import torch.nn as nn

from .feed_forward import PositionWiseFeedForward
from .multi_head_attention import MultiHeadAttention
from .add_and_norm import AddAndNorm

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_proba):
        super(TransformerEncoderBlock, self).__init__()

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.mul_head_atten_layer = MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_1 = nn.Dropout(dropout_proba)
        self.add_and_norm_layer1 = AddAndNorm(d_model)

        self.feed_forward_layer = PositionWiseFeedForward(d_model, d_ff)
        self.dropout_layer_2 = nn.Dropout(dropout_proba)
        self.add_and_norm_layer2 = AddAndNorm(d_model)

    def forward(self, x, mask):
        # x dims : (batch_size, src_seq_len, d_model)
        # mask dim : (batch_size, 1, 1, src_seq_len)

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        mul_head_atten_out = self.mul_head_atten_layer(q, k, v, mask)
        mul_head_atten_out = self.dropout_layer_1(mul_head_atten_out)
        mul_head_atten_out = self.add_and_norm_layer1(x, mul_head_atten_out)

        feed_forward_out = self.feed_forward_layer(mul_head_atten_out)
        feed_forward_out = self.dropout_layer_2(feed_forward_out)
        feed_forward_out = self.add_and_norm_layer2(mul_head_atten_out, feed_forward_out)

        return feed_forward_out


class TransformerEncoder(nn.Module):
    def __init__(self, n_blocks, n_heads, d_model, d_ff, dropout_proba=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder_blocks = nn.ModuleList([TransformerEncoderBlock(d_model, n_heads, d_ff, dropout_proba) for _ in range(n_blocks)])

    def forward(self, x, mask):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, mask)
        return x
