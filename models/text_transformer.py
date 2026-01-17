import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope_freqs(d_head, max_seq_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors.

    Reference: HuggingFace Transformers Llama implementation
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    """
    # cos/sin are (1, 1, seq_len, d_head/2), need to repeat for full d_head
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, is_causal=True):
        b, t, d = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        q, k = apply_rope(q, k, cos, sin)

        # Scaled dot-product attention (uses Flash Attention when available)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        out = out.transpose(1, 2).contiguous().view(b, t, d)
        return self.out_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = RotaryAttention(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x, cos, sin, is_causal=True):
        h = self.ln1(x)
        attn_out = self.attn(h, cos, sin, is_causal=is_causal)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.ff(h)
        return x


class TextTransformer(nn.Module):
    def __init__(self, vocab_size=4096, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

        # Precompute RoPE frequencies
        d_head = d_model // n_heads
        cos, sin = precompute_rope_freqs(d_head, max_seq_len)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def forward(self, x):
        b, t = x.shape
        h = self.token_embedding(x)

        # Get RoPE embeddings for current sequence length
        cos = self.rope_cos[:t].unsqueeze(0).unsqueeze(0)  # (1, 1, t, d_head/2)
        sin = self.rope_sin[:t].unsqueeze(0).unsqueeze(0)

        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits
