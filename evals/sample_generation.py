"""
Sample generation script using the same prompts as nanochat's base_train.py.
Tests model quality on standard completion tasks.

Usage:
    python -m evals.sample_generation --checkpoint_path runs/fineweb_edu.pt
    python -m evals.sample_generation --checkpoint_path runs/fineweb_edu.pt --max_tokens 32
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from models.text_transformer import TextTransformer


# Same prompts used in nanochat's base_train.py for consistent evaluation
SAMPLE_PROMPTS = [
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]


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
    return tokenizer


@torch.no_grad()
def generate_batch(model, input_ids, tokenizer, max_new_tokens, temperature=0.0, top_k=None, device="cuda"):
    """
    Generate tokens autoregressively (greedy by default for reproducibility).

    Args:
        model: The transformer model
        input_ids: Starting token ids (list of ints)
        tokenizer: HuggingFace tokenizer for decoding
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0 = greedy)
        top_k: Top-k sampling parameter
        device: Device to run on

    Returns:
        Full generated sequence as list of token ids
    """
    model.eval()
    tokens = list(input_ids)
    seq_len = model.max_seq_len
    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        # Truncate to max sequence length
        if len(tokens) <= seq_len:
            context = tokens
        else:
            context = tokens[-seq_len:]

        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(x)
        logits = logits[0, -1, :]

        # Sample next token
        if temperature == 0.0:
            next_token = torch.argmax(logits).item()
        elif top_k is not None and top_k > 0:
            k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, k)
            values = values / temperature
            probs = F.softmax(values, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            next_token = indices[idx].item()
        else:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        tokens.append(next_token)

        # Stop on EOS token
        if next_token == eos_token_id:
            break

    return tokens


def main():
    parser = argparse.ArgumentParser(description='Sample generation with nanochat prompts')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='meta-llama/Llama-2-7b-hf',
                        help='HuggingFace tokenizer name')
    parser.add_argument('--max_tokens', type=int, default=16,
                        help='Maximum new tokens to generate per prompt')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (0 = greedy)')
    parser.add_argument('--top_k', type=int, default=None,
                        help='Top-k sampling parameter')
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=448)
    parser.add_argument('--n_heads', type=int, default=7)
    parser.add_argument('--d_ff', type=int, default=1152)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--device', type=str, default='',
                        help='Device (auto-detected if empty)')
    parser.add_argument('--prompts', type=str, nargs='*', default=None,
                        help='Custom prompts (uses default nanochat prompts if not provided)')
    args = parser.parse_args()

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load tokenizer
    tokenizer = get_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer)

    # Load model
    model = TextTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Set up autocast for efficiency
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    autocast_ctx = torch.amp.autocast(device_type=device, dtype=dtype) if device == "cuda" else torch.inference_mode()

    # Use custom prompts or default nanochat prompts
    prompts = args.prompts if args.prompts else SAMPLE_PROMPTS

    print(f"\nSample Generation (temperature={args.temperature}, max_tokens={args.max_tokens})")
    print("=" * 60)

    for prompt in prompts:
        # Tokenize with BOS
        input_ids = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False)

        with autocast_ctx:
            output_ids = generate_batch(
                model, input_ids, tokenizer,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=device,
            )

        # Decode and print
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        print(output_text)

    print("=" * 60)


if __name__ == "__main__":
    main()
