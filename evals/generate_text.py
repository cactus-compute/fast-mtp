import argparse
import os
import sys
import torch
import torch.nn.functional as F
from models.text_transformer import TextTransformer

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(here, "..")
datasets_dir = os.path.join(project_root, "datasets")
if datasets_dir not in sys.path:
    sys.path.append(datasets_dir)
from text_lm import load_text_datasets

def build_id_to_token(vocab):
    id_to_token = {}
    for tok, idx in vocab.items():
        id_to_token[idx] = tok
    return id_to_token

def encode_prompt(prompt_tokens, vocab):
    unk_id = vocab.get("<unk>", 1)
    ids = [vocab.get(tok, unk_id) for tok in prompt_tokens]
    return ids

def decode_tokens(token_ids, id_to_token):
    tokens = [id_to_token.get(i, "<unk>") for i in token_ids]
    return " ".join(tokens)

def generate(model, prompt_ids, id_to_token, max_new_tokens, seq_len, device, greedy=True, top_k=None):
    model.eval()
    tokens = list(prompt_ids)
    for _ in range(max_new_tokens):
        if len(tokens) < seq_len:
            input_ids = tokens
        else:
            input_ids = tokens[-seq_len:]
        x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        last_logits = logits[0, -1, :]
        if top_k is not None and top_k > 0:
            values, indices = torch.topk(last_logits, top_k)
            probs = F.softmax(values, dim=-1)
            idx = indices[torch.multinomial(probs, 1).item()].item()
        elif greedy:
            idx = torch.argmax(last_logits).item()
        else:
            probs = F.softmax(last_logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
        tokens.append(idx)
    return tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["ptb", "wikitext2"])
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_vocab_size", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--prompt", type=str, default="the dog")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--top_k", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, val_ds, test_ds, vocab = load_text_datasets(args.dataset, seq_len=args.seq_len, max_vocab_size=args.max_vocab_size)
    vocab_size = len(vocab)
    id_to_token = build_id_to_token(vocab)
    model = TextTransformer(vocab_size=vocab_size, d_model=64, n_heads=4, n_layers=args.n_layers, d_ff=256, max_seq_len=args.seq_len).to(device)

    if args.checkpoint_path == "":
        default_name = f"baseline_{args.dataset}.pt"
        ckpt_path = os.path.join(project_root, "runs", default_name)
    else:
        ckpt_path = args.checkpoint_path
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)

    prompt_tokens = args.prompt.strip().split()
    prompt_ids = encode_prompt(prompt_tokens, vocab)
    generated_ids = generate(
        model,
        prompt_ids,
        id_to_token,
        max_new_tokens=args.max_new_tokens,
        seq_len=args.seq_len,
        device=device,
        greedy=args.greedy,
        top_k=args.top_k if args.top_k > 0 else None,
    )
    text = decode_tokens(generated_ids, id_to_token)
    print(text)

if __name__ == "__main__":
    main()
