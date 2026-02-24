"""Data loading and tokenization for language model training.

Uses HuggingFace datasets for data and a BPE tokenizer. We pack sequences
efficiently by concatenating documents and chunking into fixed-length sequences.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np


class TextDataset(Dataset):
    """Pre-tokenized, packed text dataset for efficient LM training.

    Instead of padding individual documents, we concatenate all tokenized text
    and split into fixed-length chunks. This ensures every token in every batch
    contributes to the loss — no wasted compute on padding.
    """

    def __init__(self, data: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        self.data = data
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        return chunk[:-1], chunk[1:]


def prepare_data(
    dataset_name: str = "roneneldan/TinyStories",
    tokenizer_name: str = "gpt2",
    seq_len: int = 1024,
    val_fraction: float = 0.01,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
):
    """Load, tokenize, and pack a dataset."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, trust_remote_code=True)

    text_column = None
    for col in ["text", "content", "document"]:
        if col in ds["train"].column_names:
            text_column = col
            break
    if text_column is None:
        raise ValueError(f"Could not find text column. Columns: {ds['train'].column_names}")

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing dataset...")
    train_data = ds["train"]
    if max_train_samples is not None:
        train_data = train_data.select(range(min(max_train_samples, len(train_data))))

    all_tokens = []
    eos_id = tokenizer.eos_token_id
    for i, example in enumerate(train_data):
        tokens = tokenizer.encode(example[text_column])
        all_tokens.extend(tokens)
        all_tokens.append(eos_id)
        if (i + 1) % 50000 == 0:
            print(f"  Tokenized {i + 1}/{len(train_data)} examples...")

    all_tokens = np.array(all_tokens, dtype=np.uint16)
    print(f"Total tokens: {len(all_tokens):,}")

    n_val = int(len(all_tokens) * val_fraction)
    rng = np.random.default_rng(seed)
    chunk_size = seq_len + 1
    n_chunks = len(all_tokens) // chunk_size
    usable = n_chunks * chunk_size
    chunks = all_tokens[:usable].reshape(n_chunks, chunk_size)
    rng.shuffle(chunks)
    shuffled = chunks.reshape(-1)

    n_train = len(shuffled) - n_val
    train_tokens = shuffled[:n_train]
    val_tokens = shuffled[n_train:n_train + n_val]

    print(f"Train tokens: {len(train_tokens):,} | Val tokens: {len(val_tokens):,}")

    train_dataset = TextDataset(train_tokens, seq_len)
    val_dataset = TextDataset(val_tokens, seq_len)
    print(f"Train samples: {len(train_dataset):,} | Val samples: {len(val_dataset):,}")

    return train_dataset, val_dataset, tokenizer


def create_dataloaders(
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    batch_size: int,
    num_workers: int = 4,
) -> tuple:
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    return train_loader, val_loader
