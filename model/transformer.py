import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin", emb.sin().unsqueeze(0).unsqueeze(0))

    def forward(self, seq_len):
        return self.cos[:, :, :seq_len], self.sin[:, :, :seq_len]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rope(T)
        q, k = apply_rope(q, k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, ffn_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CodeSeek(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_heads, n_layers, ffn_dim, max_seq_len, dropout=0.1
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ffn_dim, max_seq_len, dropout)
                for _ in range(n_layers)
            ]
        )

        self.ln_final = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.tok_embed.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape

        x = self.tok_embed(x)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=200, temperature=0.8, top_k=50):
        self.eval()

        for _ in range(max_new_tokens):
            cond = tokens[:, -self.max_seq_len :]
            logits, _ = self(cond)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == 2:
                break

        return tokens

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from config import (
        VOCAB_SIZE,
        D_MODEL,
        N_HEADS,
        N_LAYERS,
        FFN_DIM,
        MAX_SEQ_LEN,
        DROPOUT,
    )

    model = CodeSeek(
        VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, FFN_DIM, MAX_SEQ_LEN, DROPOUT
    )
    n_params = model.count_params()

    print(f"Model: CodeSeek")
    print(f"Parameters: {n_params/1e6:.1f}M")
    print(f"Layers: {N_LAYERS}")
    print(f"Hidden Dim: {D_MODEL}")
    print(f"Heads: {N_HEADS}")
    print(f"Max Seq Len: {MAX_SEQ_LEN}")

    dummy = torch.randint(0, VOCAB_SIZE, (2, 64))
    logits, loss = model(dummy, dummy)
    print(f"Input: {dummy.shape}")
    print(f"Output: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print("CodeSeek 100M is ready!")