import os
from collections import Counter
import torch
from torch.utils.data import TensorDataset

def build_vocab(train_path, max_vocab_size):
    counter = Counter()
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            counter.update(tokens)
    specials = ["<pad>", "<unk>"]
    vocab = {}
    for i, tok in enumerate(specials):
        vocab[tok] = i
    # Exclude special tokens from most_common to avoid overwriting their IDs
    for tok in specials:
        counter.pop(tok, None)
    for i, (tok, _) in enumerate(counter.most_common(max_vocab_size - len(specials)), start=len(specials)):
        vocab[tok] = i
    return vocab

def encode_file(path, vocab, unk_id=1):
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            for tok in tokens:
                ids.append(vocab.get(tok, unk_id))
    return ids

def make_lm_dataset(token_ids, seq_len):
    if len(token_ids) <= seq_len:
        x = torch.zeros((0, seq_len), dtype=torch.long)
        y = torch.zeros((0, seq_len), dtype=torch.long)
        return TensorDataset(x, y)
    ids = torch.tensor(token_ids, dtype=torch.long)
    num_seq = (len(ids) - 1) // seq_len
    ids = ids[: num_seq * seq_len + 1]
    x = ids[:-1].view(num_seq, seq_len)
    y = ids[1:].view(num_seq, seq_len)
    return TensorDataset(x, y)

def load_text_datasets(dataset_name, seq_len=64, max_vocab_size=4096):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, dataset_name)
    train_path = os.path.join(dataset_dir, "train.txt")
    val_path = os.path.join(dataset_dir, "validation.txt")
    test_path = os.path.join(dataset_dir, "test.txt")
    vocab = build_vocab(train_path, max_vocab_size)
    train_ids = encode_file(train_path, vocab)
    val_ids = encode_file(val_path, vocab)
    test_ids = encode_file(test_path, vocab)
    train_ds = make_lm_dataset(train_ids, seq_len)
    val_ds = make_lm_dataset(val_ids, seq_len)
    test_ds = make_lm_dataset(test_ids, seq_len)
    return train_ds, val_ds, test_ds, vocab
