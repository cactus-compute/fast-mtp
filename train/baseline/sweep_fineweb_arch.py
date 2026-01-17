"""
Architecture sweep for 50M parameter models on FineWeb-Edu (10M tokens).

Compares three configs:
  A: Deep-Narrow  (384d, 6 heads, 10 layers, d_ff=1024)
  B: Balanced     (448d, 7 heads, 8 layers,  d_ff=1152)
  C: Wide-Shallow (512d, 8 heads, 6 layers,  d_ff=1344)
"""

import os
import csv
import argparse
from types import SimpleNamespace
from train.baseline.train_fineweb import train_fineweb


ARCH_CONFIGS = {
    "A_deep_narrow": {
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 10,
        "d_ff": 1024,
    },
    "B_balanced": {
        "d_model": 448,
        "n_heads": 7,
        "n_layers": 8,
        "d_ff": 1152,
    },
    "C_wide_shallow": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 1344,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", type=int, default=10_000_000,
                        help="Total tokens to train on (default 10M)")
    parser.add_argument("--seq_len", type=int, default=512,
                        help="Sequence length (default 512)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Global batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for AdamW (default 3e-4)")
    parser.add_argument("--muon_lr", type=float, default=0.02,
                        help="Learning rate for Muon (default 0.02)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--subset", type=str, default="sample-10BT",
                        help="FineWeb-Edu subset")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--val_sequences", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="fast-mtp")
    parser.add_argument("--wandb_entity", type=str, default="cactuscompute-cactus-compute")
    parser.add_argument("--wandb_mode", type=str, default="online")
    parser.add_argument("--configs", type=str, nargs="+", default=None,
                        help="Specific configs to run (e.g., A_deep_narrow B_balanced)")
    args_cli = parser.parse_args()

    os.makedirs("runs", exist_ok=True)

    results_path = os.path.join("runs", "fineweb_arch_sweep.csv")
    header = [
        "config_name", "d_model", "n_heads", "n_layers", "d_ff",
        "max_tokens", "seq_len", "batch_size", "lr", "optimizer",
        "final_train_loss", "final_val_loss", "params"
    ]
    write_header = not os.path.exists(results_path)

    # Select which configs to run
    if args_cli.configs:
        configs_to_run = {k: v for k, v in ARCH_CONFIGS.items() if k in args_cli.configs}
    else:
        configs_to_run = ARCH_CONFIGS

    best_val_loss = float("inf")
    best_config = None

    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for config_name, arch in configs_to_run.items():
            print(f"\n{'='*60}")
            print(f"Running config: {config_name}")
            print(f"  d_model={arch['d_model']}, n_heads={arch['n_heads']}, "
                  f"n_layers={arch['n_layers']}, d_ff={arch['d_ff']}")
            print(f"{'='*60}\n")

            # Select appropriate LR based on optimizer
            effective_lr = args_cli.muon_lr if args_cli.optimizer == "muon" else args_cli.lr

            args = SimpleNamespace(
                max_tokens=args_cli.max_tokens,
                seq_len=args_cli.seq_len,
                subset=args_cli.subset,
                tokenizer=args_cli.tokenizer,
                val_sequences=args_cli.val_sequences,
                num_workers=args_cli.num_workers,
                d_model=arch["d_model"],
                n_heads=arch["n_heads"],
                n_layers=arch["n_layers"],
                d_ff=arch["d_ff"],
                optimizer=args_cli.optimizer,
                batch_size=args_cli.batch_size,
                lr=args_cli.lr,
                muon_lr=args_cli.muon_lr,
                weight_decay=args_cli.weight_decay,
                mixed_precision=args_cli.mixed_precision,
                log_interval=args_cli.log_interval,
                eval_interval=args_cli.eval_interval,
                wandb=args_cli.wandb,
                wandb_project=args_cli.wandb_project,
                wandb_run_name=f"arch_{config_name}_{args_cli.optimizer}_{args_cli.max_tokens // 1_000_000}M",
                wandb_entity=args_cli.wandb_entity,
                wandb_mode=args_cli.wandb_mode,
            )

            final_train_loss, final_val_loss = train_fineweb(args)

            # Estimate param count
            vocab_size = 32000  # LLaMA tokenizer
            embed_params = vocab_size * arch["d_model"]
            attn_params = 4 * arch["d_model"] ** 2
            ff_params = 3 * arch["d_model"] * arch["d_ff"]  # SwiGLU
            layer_params = attn_params + ff_params
            total_params = embed_params + arch["n_layers"] * layer_params

            row = [
                config_name,
                arch["d_model"],
                arch["n_heads"],
                arch["n_layers"],
                arch["d_ff"],
                args_cli.max_tokens,
                args_cli.seq_len,
                args_cli.batch_size,
                effective_lr,
                args_cli.optimizer,
                final_train_loss,
                final_val_loss,
                total_params,
            ]
            writer.writerow(row)
            f.flush()

            print(f"\nConfig {config_name}: val_loss={final_val_loss:.4f}, params={total_params:,}")

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_config = config_name

    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE")
    print(f"Best config: {best_config} with val_loss={best_val_loss:.4f}")
    print(f"Results saved to: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
