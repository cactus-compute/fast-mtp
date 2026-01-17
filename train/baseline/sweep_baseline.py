import warnings
warnings.warn(
    "sweep_baseline.py is deprecated. Use sweep_fineweb_arch.py instead.",
    DeprecationWarning,
    stacklevel=2
)

import os
import csv
import itertools
import argparse
from types import SimpleNamespace
from train.baseline.train_baseline import train_baseline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["ptb", "wikitext2"])
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_vocab_size", type=int, default=4096)
    args_cli = parser.parse_args()

    dataset = args_cli.dataset
    n_layers = args_cli.n_layers
    epochs = args_cli.epochs
    seq_len = args_cli.seq_len
    max_vocab_size = args_cli.max_vocab_size

    os.makedirs("runs", exist_ok=True)
    optimizers = ["adamw", "muon"]
    lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    weight_decays = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
    batch_sizes = [16, 32, 64]

    results_path = os.path.join("runs", f"baseline_sweep_results_{dataset}.csv")
    best_path = os.path.join("runs", f"baseline_sweep_best_{dataset}.csv")
    header = [
        "dataset",
        "optimizer",
        "n_layers",
        "lr",
        "weight_decay",
        "batch_size",
        "epochs",
        "seq_len",
        "max_vocab_size",
        "final_train_loss",
        "final_val_loss",
    ]
    write_header_results = not os.path.exists(results_path)
    best_val_loss = float("inf")
    best_row = None

    with open(results_path, "a", newline="") as f_results:
        writer_results = csv.writer(f_results)
        if write_header_results:
            writer_results.writerow(header)
        for optimizer, lr, wd, bs in itertools.product(optimizers, lrs, weight_decays, batch_sizes):
            args = SimpleNamespace(
                dataset=dataset,
                optimizer=optimizer,
                n_layers=n_layers,
                epochs=epochs,
                batch_size=bs,
                seq_len=seq_len,
                max_vocab_size=max_vocab_size,
                lr=lr,
                weight_decay=wd,
            )
            print("running", dataset, optimizer, "lr", lr, "weight_decay", wd, "batch_size", bs)
            final_train_loss, final_val_loss = train_baseline(args)
            row = [
                dataset,
                optimizer,
                n_layers,
                lr,
                wd,
                bs,
                epochs,
                seq_len,
                max_vocab_size,
                final_train_loss,
                final_val_loss,
            ]
            writer_results.writerow(row)
            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_row = row

    if best_row is not None:
        with open(best_path, "w", newline="") as f_best:
            writer_best = csv.writer(f_best)
            writer_best.writerow(header)
            writer_best.writerow(best_row)

if __name__ == "__main__":
    main()
