import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import math
import torch.nn as nn
import time
from torch.utils.data import DataLoader

from src.transformer import Transformer
from .data_prep import TranslationDataset, collate_fn


SRC_VOCAB_SIZE = 10000
TRG_VOCAB_SIZE = 10000
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
        lrate = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -0.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lrate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    trg_vocab_size=TRG_VOCAB_SIZE,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)


optimizer = optim.Adam(
    model.parameters(),
    lr=0,
    betas=(ADAM_BETA1, ADAM_BETA2),
    eps=ADAM_EPSILON
)

lr_scheduler = TransformerLRScheduler(optimizer, D_MODEL, WARMUP_STEPS)

criterion = CrossEntropyLoss(
    ignore_index=0,
    label_smoothing=LABEL_SMOOTHING
)

# Initialize dataloaders
train_dataset = TranslationDataset(train_src, train_trg, src_vocab, trg_vocab)
val_dataset = TranslationDataset(val_src, val_trg, src_vocab, trg_vocab)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

def train(model, iterator, optimizer, criterion, clip, lr_scheduler):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        # Shift the target for teacher forcing
        trg_input = trg[:, :-1]
        trg_output = trg[:, 1:]

        # Forward pass
        output = model(src, trg_input)

        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_output = trg_output.contiguous().view(-1)

        loss = criterion(output, trg_output)
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_loss += loss.item()

        if i%100 == 0:
            print(f'Batch: {i}, Loss: {loss.item():.3f}, LR: {current_lr:.6f}')

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)

            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:]

            output = model(src, trg_input)

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_output = trg_output.contiguous().view(-1)

            loss = criterion(output, trg_output)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

best_valid_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_loader, optimizer, criterion, CLIP, lr_scheduler)
    valid_loss = evaluate(model, val_loader, criterion)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'best-model.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    print(f'\tCurrent LR: {optimizer.param_groups[0]["lr"]:.6f}')

