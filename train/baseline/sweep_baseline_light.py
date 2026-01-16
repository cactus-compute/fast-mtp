import os
import csv
import itertools
from types import SimpleNamespace
from train.baseline.train_baseline_light import train_baseline

def main():
    os.makedirs("runs", exist_ok=True)
    optimizers = ["adamw", "muon"]
    lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    weight_decays = [0.0, 1e-4, 1e-3, 1e-2, 0.1]
    batch_sizes = [16, 32, 64]
    epochs = 5
    seq_len = 64
    vocab_size = 4096
    num_sequences = 4096
    results_path = os.path.join("runs", "baseline_sweep_results.csv")
    best_path = os.path.join("runs", "baseline_sweep_best.csv")
    header = ["optimizer", "lr", "weight_decay", "batch_size", "epochs", "final_train_loss", "final_val_loss"]
    write_header_results = not os.path.exists(results_path)
    best_val_loss = float("inf")
    best_row = None
    with open(results_path, "a", newline="") as f_results:
        writer_results = csv.writer(f_results)
        if write_header_results:
            writer_results.writerow(header)
        for optimizer, lr, wd, bs in itertools.product(optimizers, lrs, weight_decays, batch_sizes):
            args = SimpleNamespace(
                optimizer=optimizer,
                epochs=epochs,
                batch_size=bs,
                seq_len=seq_len,
                vocab_size=vocab_size,
                num_sequences=num_sequences,
                lr=lr,
                weight_decay=wd,
            )
            print("running", optimizer, "lr", lr, "weight_decay", wd, "batch_size", bs)
            final_train_loss, final_val_loss = train_baseline(args)
            row = [optimizer, lr, wd, bs, epochs, final_train_loss, final_val_loss]
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
