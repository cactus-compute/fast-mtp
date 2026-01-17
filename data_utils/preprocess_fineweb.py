"""
Preprocess FineWeb-Edu dataset and save to disk.

Run this ONCE before distributed training to avoid timeout issues:

    python -m data_utils.preprocess_fineweb --output_dir data/fineweb_edu_1024 --seq_len 1024 --num_proc 64

Then train with:

    accelerate launch ... --load_preprocessed --data_path data/fineweb_edu_1024
"""

import argparse
import os

from data_utils.fineweb_edu import get_fineweb_dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess FineWeb-Edu dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save preprocessed dataset")
    parser.add_argument("--seq_len", type=int, default=1024,
                        help="Sequence length for chunks")
    parser.add_argument("--subset", type=str, default="sample-10BT",
                        help="FineWeb-Edu subset")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Tokenizer to use")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max raw samples to load (None for all)")
    parser.add_argument("--num_proc", type=int, default=64,
                        help="Number of processes for preprocessing")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HuggingFace cache directory")
    args = parser.parse_args()

    print(f"Preprocessing FineWeb-Edu with seq_len={args.seq_len + 1} (includes +1 for x/y split)")
    print(f"Using {args.num_proc} processes")

    dataset, vocab_size = get_fineweb_dataset(
        seq_len=args.seq_len + 1,  # +1 for x/y split in FineWebEduDataset
        subset=args.subset,
        tokenizer_name=args.tokenizer,
        max_samples=args.max_samples,
        num_proc=args.num_proc,
        cache_dir=args.cache_dir,
    )

    print(f"Dataset size: {len(dataset)} chunks")
    print(f"Vocab size: {vocab_size}")

    os.makedirs(args.output_dir, exist_ok=True)
    dataset.save_to_disk(args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
