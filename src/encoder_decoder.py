import torch.nn as nn

from .transformer_encoder import TransformerEncoder
from .transformer_decoder import TransformerDecoder
from .positional_encoding import PositionalEncoding

import math

class EncoderDecoder(nn.Module):
    def __init__(self, d_model, n_blocks, src_vocab_size, trg_vocab_size, n_heads, d_ff, dropout_proba):
        super(EncoderDecoder, self).__init__()
        self.dropout_proba = dropout_proba
        self.d_model = d_model

        # Encoder
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.src_pos_embedding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(n_blocks, n_heads, d_model, d_ff, dropout_proba)


        # Decoder
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)
        self.trg_pos_embedding = PositionalEncoding(d_model)
        self.decoder = TransformerDecoder(n_blocks, n_heads, d_model, d_ff, dropout_proba)

        # Linear mapping to vocab size
        self.linear = nn.Linear(d_model, trg_vocab_size)

        self.init_with_xavier()

        self.src_embedding.weight = self.trg_embedding.weight
        self.linear.weight = self.trg_embedding.weight

    def encode(self, src_token_ids, src_mask):
        # Encoder
        src_embeddings = self.src_embedding(src_token_ids) * math.sqrt(self.d_model)
        src_embeddings = self.src_pos_embedding(src_embeddings)
        encoder_outputs = self.encoder(src_embeddings, src_mask)

        return encoder_outputs

    def decode(self, trg_token_ids, encoder_outputs, src_mask, trg_mask):
        # Decoder
        trg_embeddings = self.trg_embedding(trg_token_ids) * math.sqrt(self.d_model)
        trg_embeddings = self.trg_pos_embedding(trg_embeddings)
        decoder_outputs = self.decoder(trg_embeddings, encoder_outputs, src_mask, trg_mask)


        # Linear mapping to vocab size
        linear_out = self.linear(decoder_outputs)

        return linear_out

    def forward(self, src_token_ids, trg_token_ids, src_mask, trg_mask):
        encoder_outputs = self.encode(src_token_ids, src_mask)
        decoder_outputs = self.decode(trg_token_ids, encoder_outputs, src_mask, trg_mask)

        return decoder_outputs

    def init_with_xavier(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
