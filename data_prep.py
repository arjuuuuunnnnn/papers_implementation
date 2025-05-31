from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        trg_sentence = self.trg_sentences[idx]
        
        # Convert sentences to tensors of indices
        src_indices = [self.src_vocab[token] for token in src_sentence.split()]
        trg_indices = [self.trg_vocab[token] for token in trg_sentence.split()]
        
        # Add start and end tokens if needed
        src_indices = [self.src_vocab['<sos>']] + src_indices + [self.src_vocab['<eos>']]
        trg_indices = [self.trg_vocab['<sos>']] + trg_indices + [self.trg_vocab['<eos>']]
        
        return torch.LongTensor(src_indices), torch.LongTensor(trg_indices)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    
    # Pad sequences
    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=0, batch_first=True)
    
    return src_padded, trg_padded
