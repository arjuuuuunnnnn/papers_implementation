import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import math
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from collections import Counter

from src.transformer import Transformer
from data_prep import TranslationDataset, collate_fn

D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1
BATCH_SIZE = 64
NUM_EPOCHS = 1
WARMUP_STEPS = 4000
LABEL_SMOOTHING = 0.1
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.98
ADAM_EPSILON = 1e-9
CLIP = 1.0


class TransformerLRScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lrate = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lrate


def build_vocab(sentences, min_freq=2):
    # Special tokens
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}

    # Count word frequencies
    word_counts = Counter()
    for sentence in sentences:
        for word in sentence.split():
            word_counts[word] += 1

    # Add words that appear at least min_freq times
    idx = len(vocab)
    for word, count in word_counts.items():
        if count >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1

    return vocab


def load_data():
    """
    Returns: train_src, train_trg, val_src, val_trg, src_vocab, trg_vocab
    """
    train_src = [
        "hello world",
        "how are you",
        "good morning",
        "what is your name",
        "i am fine",
        "thank you very much",
        "see you tomorrow",
        "have a nice day",
        "where are you from",
        "i love programming"
    ]

    train_trg = [
        "bonjour monde",
        "comment allez-vous",
        "bonjour",
        "comment vous appelez-vous",
        "je vais bien",
        "merci beaucoup",
        "à demain",
        "bonne journée",
        "d'où venez-vous",
        "j'aime programmer"
    ]

    val_src = [
        "see you later",
        "thank you",
        "good night",
        "how old are you"
    ]

    val_trg = [
        "à bientôt",
        "merci",
        "bonne nuit",
        "quel âge avez-vous"
    ]

    # Build vocabs
    src_vocab = build_vocab(train_src + val_src, min_freq=1)  # min_freq=1 for small dataset
    trg_vocab = build_vocab(train_trg + val_trg, min_freq=1)

    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(trg_vocab)}")

    return train_src, train_trg, val_src, val_trg, src_vocab, trg_vocab


def create_padding_mask(seq, pad_idx=0):
    return (seq == pad_idx)


def train_epoch(model, iterator, optimizer, criterion, clip, lr_scheduler, device):
    model.train()
    epoch_loss = 0
    total_tokens = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        # Create padding mask for source
        src_mask = create_padding_mask(src)

        # Shift the target for teacher forcing
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]

        output = model(src, trg_input, mask=src_mask)

        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_output = trg_output.contiguous().view(-1)

        loss = criterion(output, trg_output)

        #count non-padding tokens for loss scaling
        non_pad_tokens = (trg_output != 0).sum().item()
        total_tokens += non_pad_tokens

        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        lr_scheduler.step()

        epoch_loss += loss.item() * non_pad_tokens

        if i % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Batch: {i:3d}, Loss: {loss.item():.4f}, LR: {current_lr:.8f}')

    return epoch_loss / total_tokens


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)

            src_mask = create_padding_mask(src)

            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]

            output = model(src, trg_input, mask=src_mask)

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_output = trg_output.contiguous().view(-1)

            loss = criterion(output, trg_output)

            non_pad_tokens = (trg_output != 0).sum().item()
            total_tokens += non_pad_tokens
            epoch_loss += loss.item() * non_pad_tokens

    return epoch_loss / total_tokens


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def main():
    train_src, train_trg, val_src, val_trg, src_vocab, trg_vocab = load_data()

    #get vocab sizes
    SRC_VOCAB_SIZE = len(src_vocab)
    TRG_VOCAB_SIZE = len(trg_vocab)

    print(f"Training samples: {len(train_src)}")
    print(f"Validation samples: {len(val_src)}")
    print(f"Source vocabulary size: {SRC_VOCAB_SIZE}")
    print(f"Target vocabulary size: {TRG_VOCAB_SIZE}")
    print("-" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #model init
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        trg_vocab_size=TRG_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("-" * 50)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0,#handled by scheduler function
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPSILON
    )

    lr_scheduler = TransformerLRScheduler(optimizer, D_MODEL, WARMUP_STEPS)

    criterion = CrossEntropyLoss(
        ignore_index=0,  #ign padding index
        label_smoothing=LABEL_SMOOTHING
    )

    # Initialize datasets and dataloaders
    train_dataset = TranslationDataset(train_src, train_trg, src_vocab, trg_vocab)
    val_dataset = TranslationDataset(val_src, val_trg, src_vocab, trg_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("=" * 50)

    #training loop
    best_valid_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, lr_scheduler, device)
        valid_loss = evaluate(model, val_loader, criterion, device)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'src_vocab': src_vocab,
                'trg_vocab': trg_vocab,
                'hyperparameters': {
                    'src_vocab_size': SRC_VOCAB_SIZE,
                    'trg_vocab_size': TRG_VOCAB_SIZE,
                    'd_model': D_MODEL,
                    'num_heads': NUM_HEADS,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT
                }
            }
            torch.save(checkpoint, 'transformer.pt')
            print("Saved new best model!")

        #perplexity
        try:
            train_ppl = math.exp(train_loss)
            valid_ppl = math.exp(valid_loss)
        except OverflowError:
            train_ppl = float('inf')
            valid_ppl = float('inf')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {train_ppl:7.3f}')
        print(f'\t Val. Loss: {valid_loss:.4f} |  Val. PPL: {valid_ppl:7.3f}')
        print(f'\tCurrent LR: {optimizer.param_groups[0]["lr"]:.8f}')
        print("-" * 50)

    print("Training completed\n")
    print(f"Best validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()
