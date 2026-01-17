"""
Streaming dataset for FineWeb-Edu using Llama tokenizer.
Yields (x, y) pairs of seq_len tokens for language modeling.
"""

import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer


class FineWebEduDataset(IterableDataset):
    """
    Streams FineWeb-Edu and yields (input, target) pairs for language modeling.

    Args:
        seq_len: Sequence length for each sample (default 4096)
        subset: HuggingFace subset name (default "sample-10BT" for 10B token sample)
        split: Dataset split (default "train")
        tokenizer_name: HuggingFace tokenizer to use (default "meta-llama/Llama-2-7b-hf")
        max_tokens: Maximum total tokens to yield (None for infinite)
        buffer_size: Number of documents to buffer for shuffling (default 1000)
    """

    def __init__(
        self,
        seq_len: int = 4096,
        subset: str = "sample-10BT",
        split: str = "train",
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
        max_tokens: int | None = None,
        buffer_size: int = 1000,
    ):
        self.seq_len = seq_len
        self.subset = subset
        self.split = split
        self.tokenizer_name = tokenizer_name
        self.max_tokens = max_tokens
        self.buffer_size = buffer_size

        # Load tokenizer (not in __init__ to avoid issues with multiprocessing)
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=True,
                trust_remote_code=True,
            )
        return self._tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def __iter__(self):
        # Get worker info for DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Get distributed rank/world_size if available
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # Compute global worker ID for sharding
        global_worker_id = rank * num_workers + worker_id
        total_workers = world_size * num_workers

        # Load dataset in streaming mode with distributed sharding
        # This ensures each worker gets a unique, non-overlapping shard
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split=self.split,
            streaming=True,
        )

        # Use HuggingFace's distributed sharding - splits the underlying data shards
        ds = ds.shard(num_shards=total_workers, index=global_worker_id)

        # Shuffle with buffer (use different seed per worker for variety)
        ds = ds.shuffle(buffer_size=self.buffer_size, seed=42 + global_worker_id)

        # Token buffer to accumulate tokens across documents
        token_buffer = []
        tokens_yielded = 0

        for example in ds:
            text = example.get("text", "")
            if not text or not text.strip():
                continue

            # Tokenize the document
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            # Yield complete sequences from buffer
            while len(token_buffer) >= self.seq_len + 1:
                # x is input, y is target (shifted by 1)
                x = torch.tensor(token_buffer[:self.seq_len], dtype=torch.long)
                y = torch.tensor(token_buffer[1:self.seq_len + 1], dtype=torch.long)

                # Remove used tokens (keep 1 for overlap)
                token_buffer = token_buffer[self.seq_len:]

                yield x, y

                tokens_yielded += self.seq_len
                if self.max_tokens is not None and tokens_yielded >= self.max_tokens:
                    return


class FineWebEduValidation(IterableDataset):
    """
    Creates a fixed validation set by taking the first N tokens from the stream.
    This ensures consistent validation across training runs.
    """

    def __init__(
        self,
        seq_len: int = 4096,
        subset: str = "sample-10BT",
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
        num_sequences: int = 100,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        self.subset = subset
        self.tokenizer_name = tokenizer_name
        self.num_sequences = num_sequences
        self.seed = seed
        self._cached_data = None
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                use_fast=True,
                trust_remote_code=True,
            )
        return self._tokenizer

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def _load_data(self):
        """Load and cache validation data."""
        if self._cached_data is not None:
            return self._cached_data

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name=self.subset,
            split="train",  # FineWeb-Edu only has train split
            streaming=True,
        )

        # Use a fixed seed for reproducible validation set
        ds = ds.shuffle(seed=self.seed, buffer_size=1000)

        token_buffer = []
        sequences = []

        for example in ds:
            text = example.get("text", "")
            if not text or not text.strip():
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.seq_len + 1 and len(sequences) < self.num_sequences:
                x = torch.tensor(token_buffer[:self.seq_len], dtype=torch.long)
                y = torch.tensor(token_buffer[1:self.seq_len + 1], dtype=torch.long)
                token_buffer = token_buffer[self.seq_len:]
                sequences.append((x, y))

            if len(sequences) >= self.num_sequences:
                break

        self._cached_data = sequences
        return self._cached_data

    def __iter__(self):
        data = self._load_data()
        for x, y in data:
            yield x, y

    def __len__(self):
        return self.num_sequences


def load_fineweb_edu(
    seq_len: int = 4096,
    max_tokens: int | None = None,
    val_sequences: int = 100,
    subset: str = "sample-10BT",
    tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
):
    """
    Load FineWeb-Edu streaming dataset for training.

    Args:
        seq_len: Sequence length (default 4096)
        max_tokens: Max tokens to train on (None for infinite)
        val_sequences: Number of validation sequences to cache
        subset: HuggingFace subset (default "sample-10BT")
        tokenizer_name: Tokenizer to use (default Llama-2)

    Returns:
        train_ds: Streaming training dataset
        val_ds: Cached validation dataset
        vocab_size: Tokenizer vocabulary size
    """
    train_ds = FineWebEduDataset(
        seq_len=seq_len,
        subset=subset,
        tokenizer_name=tokenizer_name,
        max_tokens=max_tokens,
    )

    val_ds = FineWebEduValidation(
        seq_len=seq_len,
        subset=subset,
        tokenizer_name=tokenizer_name,
        num_sequences=val_sequences,
    )

    vocab_size = train_ds.vocab_size

    return train_ds, val_ds, vocab_size
