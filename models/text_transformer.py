import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, attn_mask=None):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.ff(h)
        return x

class TextTransformer(nn.Module):
    def __init__(self, vocab_size=4096, d_model=64, n_heads=4, n_layers=2, d_ff=256, max_seq_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, x):
        b, t = x.shape
        device = x.device
        positions = torch.arange(t, device=device).unsqueeze(0).expand(b, t)
        h = self.token_embedding(x) + self.pos_embedding(positions)
        attn_mask = torch.full((t, t), float("-inf"), device=device)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        for block in self.blocks:
            h = block(h, attn_mask)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits
