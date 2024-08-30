# import os
# import random
# import wandb
# import torch
# from torch.utils.data import Dataset, Dataloader
# from torch.utils.data.sampler import Sampler
# from torch.nn.utils.rnn import pad_sequence
# from datasets import load_dataset
# from tokenizer import get_tokenizer_wordlevel, get_tokenizer_bpe

# def chunk(indices, chunk_size):
#     return torch.split(torch.tensor(indices), chunk_size)

# def pad_collate_fn(batch):
#     src_sentences, trg_sentences = [], []
#     for sample in batch:
#         src_sentences += [sample[0]]
#         trg_sentences += [sample[1]]

#     src_sentences = pad_sequence(src_sentences, batch_first=True, padding_value=0)
#     trg_sentences = pad_sequence(trg_sentences, batch_first=True, padding_value=0)

#     return src_sentences, trg_sentences

# class TranslationDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         src_encoded = self.dataset[idx]['translation_src']
#         trg_encoded = self.dataset[idx]['translation_trg']

#         return (
#                 torch.tensor(src_encoded),
#                 torch.tensor(trg_encoded),
#             )

# class CustomBatchSampler(Sampler):
#     def __init__(self, dataset, batch_size):
#         self.batch_size = batch_size
#         self.indeces = range(len(dataset))
#         self.batch_of_indeces = list(chunk(self.indeces, self.batch_size))
#         self.batch_of_indeces = [batch.tolist() for batch in self.batch_of_indeces]

#     def __iter__(self):
#         random.shuffle(self.batch_of_indeces)
#         return iter(self.batch_of_indeces)

#     def __len__(self):
#         return len(self.batch_of_indeces)

# def get_data(example_cnt):
#     data = load_dataset('wmt14', 'de-en', split='train').shuffle(sed=42)
#     data = data.select(range(example_cnt))
#     data = data.flatten()
#     data = data.rename_column('translation.de','translation_trg')
#     data = data.rename_column('translation.en', 'translation_src')

#     return data

# def preprocess_data(data, tokenizer, max_seq_len, test_proportion):
#     # Tokenize
#     def tokenize(example):
#         return {
#                 'translation_src': tokenizer.encode(example['translation_src']).ids,
#             'translation_trg': tokenizer.encode(example['translation_trg']).ids,
#         }
#     data = data.map(tokenize)

#     def sequence_length(example):
#         return {
#                 'length_src': [len(item) for item in example['translation_src']],
#                 'length_trg': [len(item) for item in example['translation_trg']],
#             }
#     data=data.map(sequence_length, batched=True, batch_size=10000)

#     def filter_long(example):
#         return example['length_src']<= max_seq_len and example['length_trg']<=max_seq_len
#     data=data.filter(filter_long)

#     # Split 
#     data=data.train_test_split(test_size=test_proportion)

#     # Sort each split by length for dynamic batching (see CustomBatchSampler)
#     data['train']=data['train'].sort('length_src', reverse=True)
#     data['test']=data['test'].sort('length_src', reverse=True)

#     return data

# def get_translation_dataloaders(
#     dataset_size,
#     vocab_size,
#     tokenizer_type,
#     tokenizer_save_pth,
#     test_proportion,
#     batch_size,
#     max_seq_len,
#     report_summary,
#     ):

#     data=get_data(dataset_size)
 
#     if tokenizer_type == 'wordlevel':
#         tokenizer=get_tokenizer_wordlevel(data, vocab_size)
#     elif tokenizer_type == 'bpe':
#         tokenizer=get_tokenizer_bpe(data, vocab_size)

#     # Save tokenizers
#     tokenizer.save(tokenizer_save_pth)

#     data=preprocess_data(data, tokenizer, max_seq_len, test_proportion)

#     if report_summary:
#         wandb.run.summary['train_len']=len(data['train'])
#         wandb.run.summary['val_len']=len(data['test'])

#     # Create pytorch datasets
#     train_ds=TranslationDataset(data['train'])
#     val_ds=TranslationDataset(data['test'])

#     # Create a custom batch sampler
#     custom_batcher_train = CustomBatchSampler(train_ds, batch_size)
#     custom_batcher_val= CustomBatchSampler(val_ds, batch_size)

#     # Create pytorch dataloaders
#     train_dl=DataLoader(train_ds, collate_fn=pad_collate_fn, batch_sampler=custom_batcher_train, pin_memory=True)
#     val_dl=DataLoader(val_ds, collate_fn=pad_collate_fn, batch_sampler=custom_batcher_val, pin_memory=True)

#     return train_dl, val_dl



import os
import random
import wandb
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torch.nn.utils.rnn import pad_sequence
from tokenizer import get_tokenizer_wordlevel, get_tokenizer_bpe, get_kannada_tokenizer

def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)

def pad_collate_fn(batch):
    src_sentences, trg_sentences = [], []
    for sample in batch:
        src_sentences += [sample[0]]
        trg_sentences += [sample[1]]

    src_sentences = pad_sequence(src_sentences, batch_first=True, padding_value=0)
    trg_sentences = pad_sequence(trg_sentences, batch_first=True, padding_value=0)

    return src_sentences, trg_sentences

class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data):
        self.src_data = src_data
        self.trg_data = trg_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.src_data[idx]),
            torch.tensor(self.trg_data[idx]),
        )

class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.indeces = range(len(dataset))
        self.batch_of_indeces = list(chunk(self.indeces, self.batch_size))
        self.batch_of_indeces = [batch.tolist() for batch in self.batch_of_indeces]

    def __iter__(self):
        random.shuffle(self.batch_of_indeces)
        return iter(self.batch_of_indeces)

    def __len__(self):
        return len(self.batch_of_indeces)

def read_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def get_data(data_dir, example_cnt=None):
    src_path = os.path.join(data_dir, 'train.en')
    trg_path = os.path.join(data_dir, 'train.kn')
    
    src_data = read_data(src_path)
    trg_data = read_data(trg_path)
    
    if example_cnt is not None:
        src_data = src_data[:example_cnt]
        trg_data = trg_data[:example_cnt]
    
    return list(zip(src_data, trg_data))

def preprocess_data(data, src_tokenizer, trg_tokenizer, max_seq_len, test_proportion):
    # Tokenize
    tokenized_data = []
    for src, trg in data:
        src_encoded = src_tokenizer.encode(src).ids
        trg_encoded = trg_tokenizer.encode(trg).input_ids
        if len(src_encoded) <= max_seq_len and len(trg_encoded) <= max_seq_len:
            tokenized_data.append((src_encoded, trg_encoded))

    # Shuffle and split the data
    random.shuffle(tokenized_data)
    split_index = int(len(tokenized_data) * (1 - test_proportion))
    train_data = tokenized_data[:split_index]
    test_data = tokenized_data[split_index:]

    # Sort each split by source sentence length for dynamic batching
    train_data.sort(key=lambda x: len(x[0]), reverse=True)
    test_data.sort(key=lambda x: len(x[0]), reverse=True)

    return {'train': train_data, 'test': test_data}

def get_translation_dataloaders(
    data_dir,
    src_vocab_size,
    src_tokenizer_type,
    trg_vocab_size,
    tokenizer_save_pth,
    test_proportion,
    batch_size,
    max_seq_len,
    report_summary,
    example_cnt=None
):
    data = get_data(data_dir, example_cnt)

    # Source (English) tokenizer
    if src_tokenizer_type == 'wordlevel':
        src_tokenizer = get_tokenizer_wordlevel([item[0] for item in data], src_vocab_size)
    elif src_tokenizer_type == 'bpe':
        src_tokenizer = get_tokenizer_bpe([item[0] for item in data], src_vocab_size)

    # Target (Kannada) tokenizer
    trg_tokenizer = get_kannada_tokenizer(trg_tokenizer_model_name)

    # Save tokenizers
    src_tokenizer.save(f"{tokenizer_save_pth}_src")
    trg_tokenizer.save_pretrained(f"{tokenizer_save_pth}_trg")

    processed_data = preprocess_data(data, src_tokenizer, trg_tokenizer, max_seq_len, test_proportion)

    if report_summary:
        wandb.run.summary['train_len'] = len(processed_data['train'])
        wandb.run.summary['val_len'] = len(processed_data['test'])

    # Create pytorch datasets
    train_ds = TranslationDataset(*zip(*processed_data['train']))
    val_ds = TranslationDataset(*zip(*processed_data['test']))

    # Create a custom batch sampler
    custom_batcher_train = CustomBatchSampler(train_ds, batch_size)
    custom_batcher_val = CustomBatchSampler(val_ds, batch_size)

    # Create pytorch dataloaders
    train_dl = DataLoader(train_ds, collate_fn=pad_collate_fn, batch_sampler=custom_batcher_train, pin_memory=True)
    val_dl = DataLoader(val_ds, collate_fn=pad_collate_fn, batch_sampler=custom_batcher_val, pin_memory=True)

    return train_dl, val_dl

# train_dataloader, val_dataloader = get_translation_dataloaders(
#     data_dir='data/',
#     src_vocab_size=30000,
#     src_tokenizer_type='bpe',
#     trg_tokenizer_model_name='ai4bharat/indic-bert',  # or any other appropriate Kannada tokenizer
#     tokenizer_save_pth='tokenizers/en_kn',
#     test_proportion=0.1,
#     batch_size=64,
#     max_seq_len=128,
#     report_summary=True,
#     example_cnt=None  # Set to a number if you want to limit the dataset size
# )
