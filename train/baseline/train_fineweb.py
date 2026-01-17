"""
Training script for FineWeb-Edu with streaming data and token-based training.
"""

import argparse
import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from models.text_transformer import TextTransformer
from optimizers.adamw import build_adamw_optimizer
from optimizers.muon import build_muon_optimizer
from data_utils.fineweb_edu import load_fineweb_edu


def _wandb_init(args):
    if not getattr(args, "wandb", False):
        return None
    try:
        import wandb
    except BaseException as e:
        print("warning: wandb unavailable:", repr(e))
        return None
    try:
        run_name = args.wandb_run_name
        if run_name is None or str(run_name).strip() == "":
            run_name = f"fineweb_{args.max_tokens // 1_000_000}M"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config=vars(args),
            mode=args.wandb_mode,
        )
    except BaseException as e:
        print("warning: wandb.init failed:", repr(e))
        return None
    return wandb


def evaluate(model, dataloader, vocab_size, accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            gathered_loss = accelerator.gather(loss.unsqueeze(0))
            losses.extend(gathered_loss.cpu().tolist())
    model.train()
    if len(losses) == 0:
        return 0.0
    return sum(losses) / len(losses)


def train_fineweb(args):
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    wandb = _wandb_init(args) if accelerator.is_main_process else None

    # Load streaming dataset
    train_ds, val_ds, vocab_size = load_fineweb_edu(
        seq_len=args.seq_len,
        max_tokens=args.max_tokens,
        val_sequences=args.val_sequences,
        subset=args.subset,
        tokenizer_name=args.tokenizer,
    )

    # Adjust batch size per GPU
    per_device_batch_size = args.batch_size // accelerator.num_processes

    # For IterableDataset, we don't shuffle in DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=per_device_batch_size,
        num_workers=0,  # Validation is cached, no need for workers
    )

    # Scale model size for larger sequences
    d_ff = args.d_ff if args.d_ff is not None else args.d_model * 4
    model = TextTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=d_ff,
        max_seq_len=args.seq_len,
    )

    # Calculate total steps from token budget
    # Each step processes batch_size * seq_len tokens across all GPUs
    tokens_per_step = args.batch_size * args.seq_len
    total_steps = args.max_tokens // tokens_per_step
    # Multiply by num_processes since each GPU calls scheduler.step() independently
    total_steps_for_scheduler = total_steps * accelerator.num_processes
    warmup_steps = int(0.1 * total_steps_for_scheduler)

    if accelerator.is_main_process:
        print(f"Training for {total_steps} steps ({args.max_tokens:,} tokens)")
        print(f"Tokens per step: {tokens_per_step:,}")
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.optimizer == "adamw":
        lr = args.lr
        optimizer, scheduler = build_adamw_optimizer(
            model, lr, args.weight_decay, warmup_steps, total_steps_for_scheduler
        )
    else:
        lr = getattr(args, "muon_lr", 0.02)
        optimizer, scheduler = build_muon_optimizer(
            model, lr, args.weight_decay, warmup_steps, total_steps_for_scheduler
        )

    if accelerator.is_main_process:
        print(f"Optimizer: {args.optimizer}, LR: {lr}")
        print(f"Scheduler: warmup_steps={warmup_steps}, total_steps_for_scheduler={total_steps_for_scheduler}")

    # Prepare for distributed training
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    if accelerator.is_main_process:
        os.makedirs("runs", exist_ok=True)

    # Training loop
    step = 0
    tokens_seen = 0
    running_loss = 0.0
    log_interval = args.log_interval
    eval_interval = args.eval_interval

    train_losses = []
    val_losses = []

    pbar = tqdm(total=total_steps, disable=not accelerator.is_main_process, desc="Training")

    for x, y in train_loader:
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        step += 1
        tokens_seen += tokens_per_step

        # Logging
        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            train_losses.append(avg_loss)

            if accelerator.is_main_process:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "tokens": f"{tokens_seen:,}"})
                if wandb is not None:
                    wandb.log({
                        "train_loss": avg_loss,
                        "tokens_seen": tokens_seen,
                        "lr": current_lr,
                    }, step=step)

            running_loss = 0.0

        # Evaluation
        if step % eval_interval == 0:
            val_loss = evaluate(model, val_loader, vocab_size, accelerator)
            val_losses.append(val_loss)

            if accelerator.is_main_process:
                print(f"\nStep {step} | Val loss: {val_loss:.4f}")
                if wandb is not None:
                    wandb.log({"val_loss": val_loss, "step": step}, step=step, commit=True)

        pbar.update(1)

        if step >= total_steps:
            break

    pbar.close()

    # Final evaluation
    final_val_loss = evaluate(model, val_loader, vocab_size, accelerator)
    final_train_loss = train_losses[-1] if train_losses else 0.0

    if accelerator.is_main_process:
        print(f"\nTraining complete. Final val loss: {final_val_loss:.4f}")
        if wandb is not None:
            wandb.finish()

    # Save model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join("runs", "fineweb_edu.pt"))

        # Save metrics
        metrics_path = os.path.join("runs", "fineweb_metrics.csv")
        header = [
            "max_tokens", "optimizer", "n_layers", "d_model", "lr",
            "weight_decay", "batch_size", "seq_len", "final_train_loss", "final_val_loss"
        ]
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([
                args.max_tokens, args.optimizer, args.n_layers, args.d_model,
                args.lr, args.weight_decay, args.batch_size, args.seq_len,
                final_train_loss, final_val_loss
            ])

    return final_train_loss, final_val_loss


def main():
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--max_tokens", type=int, default=100_000_000,
                        help="Total tokens to train on (default 100M)")
    parser.add_argument("--seq_len", type=int, default=4096,
                        help="Sequence length (default 4096)")
    parser.add_argument("--subset", type=str, default="sample-10BT",
                        help="FineWeb-Edu subset (default sample-10BT)")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="HuggingFace tokenizer name")
    parser.add_argument("--val_sequences", type=int, default=100,
                        help="Number of validation sequences to cache")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")

    # Model args
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=None,
                        help="FFN hidden dim (default: 4*d_model)")

    # Training args
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Global batch size (default 8, smaller due to seq_len=4096)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate for AdamW (default 3e-4)")
    parser.add_argument("--muon_lr", type=float, default=0.02,
                        help="Learning rate for Muon (default 0.02, only used with --optimizer muon)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])

    # Logging args
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Steps between logging")
    parser.add_argument("--eval_interval", type=int, default=500,
                        help="Steps between evaluation")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="fast-mtp")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default="cactuscompute-cactus-compute")
    parser.add_argument("--wandb_mode", type=str, default="online")

    args = parser.parse_args()
    train_fineweb(args)


if __name__ == "__main__":
    main()
