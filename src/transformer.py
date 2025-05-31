import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model=512, num_heads=8, num_layers=6, dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)

        self.encoder = Encoder(num_layers, d_model, num_heads, dropout, max_len)
        self.decoder = Decoder(num_layers, d_model, dropout, num_heads, max_len)

        self.linear = nn.Linear(d_model, trg_vocab_size)

        #if probabilities needed
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, trg, mask=None):
        src_embedding = self.src_embedding(src)
        trg_embedding = self.trg_embedding(trg)

        encoder_output = self.encoder(x=src_embedding, mask=mask)
        decoder_output = self.decoder(x=trg_embedding, encoder_output=encoder_output, mask=None)#causal mask is handled in decoder so no need to mention here
        output = self.linear(decoder_output)
        # softmax_output = self.softmax(output)
        return output

    def encode(self, src, mask=None):
        src_embedded = self.src_embedding(src)
        return self.encoder(src_embedded, mask)

    def decode(self, trg, encoder_output, mask=None):
        trg_embedded = self.trg_embedding(trg)
        return self.decoder(trg_embedded, encoder_output, mask)

