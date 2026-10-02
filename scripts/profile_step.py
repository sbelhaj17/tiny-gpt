"""Where a training step's time goes on the GPU, and why bf16 barely helps.

    python scripts/profile_step.py

Three measurements, each the mean over repeated calls after a few warmup calls:
1. matmul throughput in fp32, fp16 and bf16, at the shape of the MLP's first
   layer (32 x 256 tokens, 384 -> 1536) and at a large square shape;
2. one layer's attention (32 sequences, 6 heads, 256 tokens, 64 dims),
   forward and backward, with PyTorch's fused kernel and written out;
3. a whole step of the 6-layer model split into forward, backward, and
   gradient clipping plus AdamW.
"""

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # the repo root
from tinygpt.data import Batches  # noqa: E402
from tinygpt.model import GPT, matmul_attention  # noqa: E402
from train import TrainConfig, autocast, make_optimizer, model_config, sync  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def mean_ms(fn, n=20, warmup=3):
    for _ in range(warmup):
        fn()
    sync(DEVICE)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    sync(DEVICE)
    return (time.perf_counter() - t) / n * 1000


def matmuls():
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        out = []
        for m, k, n in [(8192, 384, 1536), (4096, 4096, 4096)]:
            a = torch.randn(m, k, device=DEVICE, dtype=dtype)
            b = torch.randn(k, n, device=DEVICE, dtype=dtype)
            ms = mean_ms(lambda: a @ b, n=30)
            out.append(f"{m}x{k}x{n}: {2 * m * k * n / ms / 1e9:.2f} TFLOP/s")
        print(f"matmul {str(dtype)[6:]:8s}  " + ", ".join(out))


def attention():
    for dtype in [torch.float32, torch.bfloat16]:
        q, k, v = (torch.randn(32, 6, 256, 64, device=DEVICE, dtype=dtype, requires_grad=True) for _ in range(3))
        kernels = {
            "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
            "matmul": lambda: matmul_attention(q, k, v),
        }
        for name, f in kernels.items():
            fwd = mean_ms(f)
            both = mean_ms(lambda: f().sum().backward())
            print(f"attention {str(dtype)[6:]:8s} {name:6s}  forward {fwd:5.1f} ms, forward+backward {both:5.1f} ms")


def step():
    for dtype, attn in [("fp32", "matmul"), ("bf16", "matmul"), ("bf16", "sdpa")]:
        cfg = TrainConfig(dtype=dtype, attn=attn, device=DEVICE)
        torch.manual_seed(0)
        model = GPT(model_config(cfg, 4096)).to(DEVICE)
        opt = make_optimizer(model, cfg)
        x, y = Batches("data/train.bin", cfg.batch_size, cfg.block_size, DEVICE).next()

        def forward():
            with autocast(cfg):
                return model(x, y)[1]

        def update():
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        f = mean_ms(forward)
        fb = mean_ms(lambda: forward().backward())
        u = mean_ms(update)
        print(f"step {dtype} {attn:6s}  forward {f:4.0f} ms, backward {fb - f:4.0f} ms, clip+AdamW {u:3.0f} ms")


if __name__ == "__main__":
    print(f"device: {DEVICE}, torch {torch.__version__}")
    matmuls()
    attention()
    step()
