import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from config import configs
from learner import Learner
from scheduler import CustomScheduler
from dataset import get_translation_dataloaders
from callbacks import CheckpointSaver, MoveToDeviceCallback, TrackLoss, TrackExample, TrackBleu
from src.translation_transformer import TranslationTransformer
from utils.logconf import logging
import numpy as np
import random


config_name='cpu_configs' # CHANGE
wandb.init(config=configs[config_name],project="attention-is-all-you-need-paper", entity="bkoch4142")

# Configure Logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Seed the Random Number Generators
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

os.makedirs('tokenizers', exist_ok=True)


class TrainingApp:
    def __init__(self):

        log.info('----- Training Started -----')

        # Device handling
        if wandb.config.DEVICE=='gpu':
            if not torch.cuda.is_available():
                raise ValueError('GPU is not available.')
            self.device = 'cuda'
            log.info(f'Device name is {torch.cuda.get_device_name()}')
        else:
            log.info(f'Device name is CPU')
            self.device='cpu'

    def main(self):
        train_dl, val_dl, src_tokenizer, trg_tokenizer = get_translation_dataloaders(
            data_dir='data/',
            src_vocab_size=32000,
            trg_tokenizer_model_name=wandb.config.MODEL_NAME_KANNADA,
            tokenizer_save_pth='tokenizers/en_kn',
            test_proportion=wandb.config.TEST_PROPORTION,
            batch_size=wandb.config.BATCH_SIZE,
            max_seq_len=wandb.config.MAX_SEQ_LEN,
            report_summary=True,
            example_cnt=None  # Set a number if you want to limit the dataset size
        )

        model = TranslationTransformer(
            d_model=wandb.config.D_MODEL,
            n_blocks=wandb.config.N_BLOCKS,
            src_vocab_size=src_tokenizer.get_vocab_size(),
            trg_vocab_size=len(trg_tokenizer),
            n_heads=wandb.config.N_HEADS,
            d_ff=wandb.config.D_FF,
            dropout_proba=wandb.config.DROPOUT_PROBA
        )

        loss_func = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1, reduction='mean')

        optimizer = optim.Adam(model.parameters(), betas=wandb.config.BETAS, eps=wandb.config.EPS)
        scheduler = CustomScheduler(optimizer, wandb.config.D_MODEL, wandb.config.N_WARMUP_STEPS)
        
        cbs = [
            MoveToDeviceCallback(),
            TrackLoss(),
            TrackExample(),
            TrackBleu(),
            CheckpointSaver(epoch_cnt=wandb.config.MODEL_SAVE_EPOCH_CNT,),
        ]
        
        wandb.watch(model, log_freq=1000)
        learner = Learner(model,
                          train_dl,
                          val_dl,
                          loss_func,
                          cbs,
                          optimizer,
                          scheduler,
                          self.device)

        learner.fit(wandb.config.EPOCHS)
        
if __name__ == "__main__":
    TrainingApp().main()
