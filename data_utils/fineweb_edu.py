"""
Map-style dataset for FineWeb-Edu using Llama tokenizer.
Fully loads and preprocesses the dataset for proper distributed training with DistributedSampler.
"""

import os
import itertools
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


def get_tokenizer(tokenizer_name: str = "meta-llama/Llama-2-7b-hf"):
    """Load tokenizer and ensure it has required special tokens."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.bos_token is None:
        if tokenizer.cls_token is None:
            raise AttributeError(f'Tokenizer must have a bos_token or cls_token: {tokenizer}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        if tokenizer.sep_token is None:
            raise AttributeError(f'Tokenizer must have a eos_token or sep_token: {tokenizer}')
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer


def get_fineweb_dataset(
    seq_len: int = 1024,
    subset: str = "sample-10BT",
    split: str = "train",
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    max_samples: int | None = None,
    num_proc: int = 8,
    cache_dir: str | None = None,
    data_path: str = "HuggingFaceFW/fineweb-edu",
    load_preprocessed: bool = False,
):
    """
    Load and preprocess FineWeb-Edu dataset into fixed-length chunks.

    Args:
        seq_len: Sequence length for each sample
        subset: HuggingFace subset name (default "sample-10BT" for 10B token sample)
        split: Dataset split (default "train")
        tokenizer_name: HuggingFace tokenizer to use
        max_samples: Maximum number of raw samples to load (None for all)
        num_proc: Number of processes for preprocessing
        cache_dir: Cache directory for dataset
        data_path: Path to dataset (HF repo or local directory)
        load_preprocessed: Whether to load preprocessed data from disk (data_path)

    Returns:
        Dataset with 'input_ids' ready for language modeling
    """
    if load_preprocessed:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Preprocessed dataset not found at {data_path}")
        dataset = load_from_disk(data_path)
        # If it's a DatasetDict, pick the split
        if hasattr(dataset, "keys") and split in dataset:
            dataset = dataset[split]
        # Vocabulary size is still needed from tokenizer
        tokenizer = get_tokenizer(tokenizer_name)
        return dataset.with_format('torch'), tokenizer.vocab_size

    tokenizer = get_tokenizer(tokenizer_name)
    BOS = tokenizer.bos_token_id
    EOS = tokenizer.eos_token_id

    # Load dataset (non-streaming for proper distributed training)
    if max_samples is not None:
        split_str = f"{split}[:{max_samples}]"
    else:
        split_str = split

    dataset = load_dataset(
        data_path,
        name=subset,
        split=split_str,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )

    def preprocess_and_tokenize(examples):
        """Tokenize text without special tokens (we add them during chunking)."""
        texts = examples["text"]
        tokens = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        # Add EOS at end of each document
        tokens = {'input_ids': [t + [EOS] for t in tokens['input_ids']]}
        return tokens

    tokenized_dataset = dataset.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        desc="Tokenizing",
    )

    def group_texts(examples):
        """Concatenate all texts and split into fixed-length chunks with BOS/EOS."""
        concatenated = list(itertools.chain(*examples['input_ids']))
        total_length = len(concatenated)

        # Reserve space for BOS at start of each chunk
        # (EOS tokens are already in the stream from document boundaries)
        chunk_content_size = seq_len - 1  # -1 for BOS
        total_length = (total_length // chunk_content_size) * chunk_content_size

        result = {'input_ids': []}
        for i in range(0, total_length, chunk_content_size):
            chunk = [BOS] + concatenated[i:i + chunk_content_size]
            result['input_ids'].append(chunk)

        return result

    chunked_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Chunking",
    )

    chunked_dataset = chunked_dataset.with_format('torch')

    return chunked_dataset, tokenizer.vocab_size


class FineWebEduDataset(Dataset):
    """
    Map-style dataset wrapper for FineWeb-Edu.
    Returns (input, target) pairs for language modeling.
    """

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        tokens = self.dataset[idx]['input_ids']
        # x is input (all but last), y is target (all but first)
        x = tokens[:-1]
        y = tokens[1:]
        return x, y


def cycle_loader(dataloader, sampler=None):
    """Infinite iterator over dataloader, resetting sampler epoch for shuffling."""
    import numpy as np
    while True:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def load_fineweb_edu(
    seq_len: int = 1024,
    max_tokens: int | None = None,
    val_sequences: int = 100,
    subset: str = "sample-10BT",
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    num_workers: int = 4,
    batch_size: int = 32,
    distributed: bool = True,
    cache_dir: str | None = None,
    num_proc: int = 8,
    data_path: str = "HuggingFaceFW/fineweb-edu",
    load_preprocessed: bool = False,
):
    """
    Load FineWeb-Edu dataset with proper distributed training support.

    Args:
        seq_len: Sequence length (default 1024)
        max_tokens: Approximate max tokens to load (None for full dataset)
        val_sequences: Number of validation sequences
        subset: HuggingFace subset (default "sample-10BT")
        tokenizer_name: Tokenizer to use (default Llama-2)
        num_workers: DataLoader workers
        batch_size: Per-device batch size
        distributed: Whether to use DistributedSampler
        cache_dir: Cache directory for dataset
        num_proc: Number of processes for preprocessing
        data_path: Path to dataset (HF repo or local directory)
        load_preprocessed: Whether to load preprocessed data from disk (data_path)

    Returns:
        train_loader: Training dataloader (infinite iterator)
        val_loader: Validation dataloader (infinite iterator)
        vocab_size: Tokenizer vocabulary size
        train_sampler: Training sampler (for epoch setting)
    """
    # Estimate number of samples to load based on max_tokens
    # Each sample is ~seq_len tokens, so max_samples ~ max_tokens / seq_len
    if max_tokens is not None:
        # Load extra samples to account for chunking overhead
        max_samples = int(max_tokens / seq_len * 1.5)
    else:
        max_samples = None

    # Load and preprocess training data
    train_hf_dataset, vocab_size = get_fineweb_dataset(
        seq_len=seq_len + 1,  # +1 because we split into x[:-1], y[1:]
        subset=subset,
        split="train",
        tokenizer_name=tokenizer_name,
        max_samples=max_samples,
        num_proc=num_proc,
        cache_dir=cache_dir,
        data_path=data_path,
        load_preprocessed=load_preprocessed,
    )

    train_dataset = FineWebEduDataset(train_hf_dataset)

    # For validation, use a small fixed subset from the end of the dataset
    # to avoid overlap with training
    val_start = max(0, len(train_hf_dataset) - val_sequences)
    val_hf_dataset = train_hf_dataset.select(range(val_start, len(train_hf_dataset)))
    val_dataset = FineWebEduDataset(val_hf_dataset)

    # Exclude validation samples from training
    train_hf_dataset = train_hf_dataset.select(range(val_start))
    train_dataset = FineWebEduDataset(train_hf_dataset)

    # Create samplers for distributed training
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(train_sampler is None),
        drop_last=True,  # Ensure all batches are same size for distributed
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,  # Validation is small, no need for workers
        pin_memory=True,
        shuffle=False,
        drop_last=True,
    )

    # Wrap in infinite iterators
    train_loader = cycle_loader(train_loader, train_sampler)
    val_loader = cycle_loader(val_loader, val_sampler)

    return train_loader, val_loader, vocab_size, train_sampler
