import torch
import torch.nn as nn

from encoder_decoder import EncoderDecoder

class TranslationTransformer(nn.Module):
    def __init__(self, d_model, n_blocks, src_vocab_size, trg_vocab_size, n_heads, d_ff, dropout_proba):
        super(TranslationTransformer, self).__init__()

        self.encoder_decoder = EncoderDecoder(
                d_model,
                n_blocks,
                src_vocab_size,
                trg_vocab_size,
                n_heads,
                d_ff,
                dropout_proba
            )
    def _get_pad_mask(self, token_ids, pad_idx=0):
        pad_mask = (token_ids != pad_idx).unsqueeze(-2)
        return pad_mask.unsqueeze(1)

    def _get_lookahed_mask(self, token_ids):
        sz_b, len_s = token_ids.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=token_ids.device), diagonal=1)).bool()
        return subsequent_mask.unsqueeze(1)

    def forward(self, src_token_ids, trg_token_ids):
        # Since trg_token_ids contains both [BOS] and [SOS] tokens
        # we need to remove the [EOS] token when using it as input to the decoder.
        # Similarly we remove the [BOS] token when we use it as y to calculate loss,
        # which also makes y and y_pred shapes match.

        # Removing [EOS] token
        trg_token_ids=trg_token_ids[:, :-1]

        src_mask = self._get_pad_mask(src_token_ids) # (batch_size, 1, 1, src_seq_len)
        trg_mask = self._get_pad_mask(trg_token_ids) & self._get_lookahead_mask(trg_token_ids)  # (batch_size, 1, trg_seq_len, trg_seq_len)

        return self.transformer_encoder_decoder(src_token_ids, trg_token_ids, src_mask, trg_mask)


    def preprocess(self, sentence, tokenizer):
        device = next(self.parameters()).device

        src_token_ids=tokenizer.encode(sentence).ids
        src_token_ids=torch.tensor(src_token_ids, dtype=torch.long).to(device)
        src_token_ids=src_token_ids.unsqueeze(0) # To batch format

        return src_token_ids
