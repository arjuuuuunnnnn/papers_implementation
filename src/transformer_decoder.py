import torch.nn as nn
import os

from multi_head_attention import MultiHeadAttention
from add_and_norm import AddAndNorm
from feed_forward import PositionWiseFeedForward

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout_proba):
        super(TransformerDecoderBlock, self).__init__()

        self.W_q_1 = nn.Linear(d_model, d_model)
        self.W_k_1 = nn.Linear(d_model, d_model)
        self.W_v_1 = nn.Linear(d_model, d_model)

        self.mul_head_atten_layer_1 = MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_1 = nn.Dropout(dropout_proba)
        self.add_and_norm_1 = AddAndNorm(d_model)

        self.W_q_2 = nn.Linear(d_model, d_model)
        self.W_k_2 = nn.Linear(d_model, d_model)
        self.W_v_2 = nn.Linear(d_model, d_model)

        self.mul_head_atten_layer_2 = MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_2 = nn.Dropout(dropout_proba)
        self.add_and_norm_2 = AddAndNorm(d_model)

        self.feed_forward_layer = PositionWiseFeedForward(d_model, d_ff)
        self.dropout_layer_3=nn.Dropout(dropout_proba)
        self.add_and_norm_3 = AddAndNorm(d_model)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        # x dims: (batch_size, trg_seq_len, d_model)
        # encoder_output dims: (batch_size, src_seq_len, d_model)
        # src_mask dim: (batch_size, 1, 1, src_seq_len)
        # trg_mask dim: (batch_size, 1, trg_seq_len, trg_seq_len)

        q_1 = self.W_q_1(x) # (batch_size, trg_seq_len, d_model)
        k_1 = self.W_k_1(x) # (batch_size, trg_seq_len, d_model)
        v_1 = self.W_v_1(x) # (batch_size, trg_seq_len, d_model)

        mul_head_atten_1_out = self.mul_head_atten_layer_1(q_1, k_1, v_1, trg_mask)
        mul_head_atten_1_out = self.dropout_layer_1(mul_head_atten_1_out)
        mul_head_atten_1_out = self.add_and_norm_1(mul_head_atten_1_out, x)
        

        q_2 = self.W_q_2(mul_head_atten_1_out)
        k_2 = self.W_k_2(encoder_output)
        v_2 = self.W_v_2(encoder_output)

        mul_head_atten_2_out = self.mul_head_atten_layer_2(q_2, k_2, v_2, src_mask)
        mul_head_atten_2_out = self.dropout_layer_2(mul_head_atten_layer_2)
        mul_head_atten_2_out = self.add_and_norm_2(mul_head_atten_2_out, mul_head_atten_1_out)

        feed_forward_out = self.feed_forward_layer(mul_head_atten_2_out)
        feed_forward_out = self.dropout_layer_3(feed_forward_out)
        feed_forward_out = self.add_and_norm_3(feed_forward_out, mul_head_atten_2_out)

        return feed_forward_out

class TransformerDecoder(nn.Module):
    def __init__(self, n_blocks, d_models, d_ff, dropout_proba):
        super(TransformerDecoder, self).__init__()

        self.decoder_blocks = nn.ModuleList([TransformerDecoderBlock(d_model, n_heads, d_ff, dropout_proba) for _ in range(n_blocks)])

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, src_mask, trg_mask)
        return x

