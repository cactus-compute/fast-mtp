import argparse
import os
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.text_transformer import TextTransformer
from optimizers.adamw import build_adamw_optimizer
from optimizers.muon import build_muon_optimizer

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
            run_name = "baseline_light"
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

def create_dataset(num_sequences=10000, seq_len=64, vocab_size=4096):
    tokens = torch.randint(0, vocab_size, (num_sequences, seq_len + 1))
    x = tokens[:, :seq_len]
    y = tokens[:, 1:seq_len + 1]
    return TensorDataset(x, y)

def evaluate(model, dataloader, vocab_size, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            losses.append(float(loss.item()))
    model.train()
    if len(losses) == 0:
        return 0.0
    return sum(losses) / len(losses)

def _derive_val_plot_path(train_plot_path, default_val_path):
    if train_plot_path is None or str(train_plot_path).strip() == "":
        return default_val_path
    root, ext = os.path.splitext(train_plot_path)
    if ext == "":
        ext = ".png"
    return root + "_val" + ext

def train_baseline(args):
    wandb = _wandb_init(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = create_dataset(args.num_sequences, args.seq_len, args.vocab_size)
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model = TextTransformer(vocab_size=args.vocab_size, max_seq_len=args.seq_len).to(device)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    if args.optimizer == "adamw":
        optimizer, scheduler = build_adamw_optimizer(model, args.lr, args.weight_decay, warmup_steps, total_steps)
    else:
        optimizer, scheduler = build_muon_optimizer(model, args.lr, args.weight_decay, warmup_steps, total_steps)
    os.makedirs("runs", exist_ok=True)
    final_train_loss = None
    final_val_loss = None
    train_losses_by_epoch = []
    val_losses_by_epoch = []
    for epoch in range(args.epochs):
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.item()))
        if len(losses) == 0:
            train_loss = 0.0
        else:
            train_loss = sum(losses) / len(losses)
        val_loss = evaluate(model, val_loader, args.vocab_size, device)
        final_train_loss = train_loss
        final_val_loss = val_loss
        train_losses_by_epoch.append(float(train_loss))
        val_losses_by_epoch.append(float(val_loss))
        print("epoch", epoch, "train_loss", float(train_loss), "val_loss", float(val_loss))
        if wandb is not None:
            wandb.log({"train_loss": float(train_loss), "val_loss": float(val_loss)}, step=int(epoch + 1))
    torch.save(model.state_dict(), os.path.join("runs", "baseline_model.pt"))
    metrics_path = os.path.join("runs", "baseline_metrics.csv")
    header = ["optimizer", "lr", "weight_decay", "batch_size", "epochs", "final_train_loss", "final_val_loss"]
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([args.optimizer, args.lr, args.weight_decay, args.batch_size, args.epochs, final_train_loss, final_val_loss])

    if args.plot_loss:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            xs = list(range(1, len(train_losses_by_epoch) + 1))
            plt.figure(figsize=(8, 4))
            plt.plot(xs, train_losses_by_epoch)
            plt.xlabel("epoch")
            plt.ylabel("train_loss")
            plt.title("Baseline(light) train loss")
            out_path = args.plot_path
            if out_path is None or str(out_path).strip() == "":
                out_path = os.path.join("runs", "baseline_light_train_loss.png")
            plt.tight_layout()
            plt.savefig(out_path)
            plt.close()
            print("saved loss plot to", out_path)

            if len(val_losses_by_epoch) > 0:
                plt.figure(figsize=(8, 4))
                plt.plot(xs, val_losses_by_epoch)
                plt.xlabel("epoch")
                plt.ylabel("val_loss")
                plt.title("Baseline(light) val loss")
                val_out_path = _derive_val_plot_path(out_path, os.path.join("runs", "baseline_light_val_loss.png"))
                plt.tight_layout()
                plt.savefig(val_out_path)
                plt.close()
                print("saved val loss plot to", val_out_path)
        except Exception as e:
            print("warning: failed to save loss plot:", repr(e))
    return final_train_loss, final_val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--num_sequences", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--plot_loss", action="store_true", help="Save train loss vs epoch plot as a PNG.")
    parser.add_argument("--plot_path", type=str, default=None, help="Optional output path for --plot_loss.")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Just Enough Learning (JEL)")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default="cactuscompute-cactus-compute")
    parser.add_argument("--wandb_mode", type=str, default="online")
    args = parser.parse_args()
    train_baseline(args)

if __name__ == "__main__":
    main()
