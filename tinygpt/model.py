import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 4096
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    pos_emb: str = "rope"  # "rope" or "learned"
    attn: str = "sdpa"  # "sdpa" or "naive"; the naive path is the reference the tests check sdpa against


def naive_attention(q, k, v):
    T = q.size(-2)
    att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    future = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    att = att.masked_fill(future, float("-inf"))
    return att.softmax(dim=-1) @ v


def rope_tables(head_dim, max_len, base=10000.0):
    # One rotation frequency per pair of channels, from 1 down to about 1/base.
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # Channel i is paired with channel i + head_dim/2 (the GPT-NeoX layout) and
    # each pair is rotated by an angle proportional to the position. The dot
    # product of a rotated query and key then depends only on their distance.
    x1, x2 = x.chunk(2, dim=-1)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.type_as(x)  # cos/sin are fp32; keep q and k in the autocast dtype


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.attn = cfg.attn
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, rope=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if self.attn == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            y = naive_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln1(x), rope)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos_emb == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        elif cfg.pos_emb == "rope":
            self.pos_emb = None
            cos, sin = rope_tables(cfg.n_embd // cfg.n_head, cfg.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            raise ValueError(f"unknown pos_emb {cfg.pos_emb!r}")
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # Weight tying: the output layer scores each token by its dot product
        # with that token's input embedding.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2's trick: each block adds two outputs to the residual stream, so
        # shrink those projections to keep the stream's variance from growing
        # with depth.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        # parameters() yields the tied embedding once
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} tokens, block size is {self.cfg.block_size}"
        x = self.tok_emb(idx)
        rope = None
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        else:
            rope = (self.rope_cos[:T], self.rope_sin[:T])
        for block in self.blocks:
            x = block(x, rope)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        # The softmax over the vocabulary is done in fp32 even under autocast.
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :].float()
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None and (nxt == stop_token).all():
                break
        return idx
