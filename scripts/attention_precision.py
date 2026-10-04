"""How far bf16 attention lands from fp32, written out against sdpa.

    python scripts/attention_precision.py

Under bf16 autocast the written-out version multiplies queries and keys in
bf16, so the attention scores are bf16; autocast then runs the softmax in fp32
and the second matmul in bf16 again. sdpa takes bf16 inputs too. The inputs
here need no gradients, so on MPS sdpa runs MPS's own attention kernel, not the
fp32 fallback it takes in training (results/sdpa_dispatch.txt). This compares
both to the same attention in fp32, on five sets of random inputs of one
layer's shape (8 sequences, 6 heads, 256 tokens, 64 dims), and prints the
largest and the mean absolute error.
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # the repo root
from tinygpt.model import matmul_attention  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def errors(seed):
    torch.manual_seed(seed)
    q, k, v = (torch.randn(8, 6, 256, 64, device=DEVICE) for _ in range(3))
    reference = matmul_attention(q, k, v)
    kernels = {
        "matmul": lambda: matmul_attention(q, k, v),
        "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True),
    }
    out = {}
    for name, f in kernels.items():
        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            y = f()
        err = (y.float() - reference).abs()
        out[name] = (err.max().item(), err.mean().item())
    return out


def main():
    print(f"device {DEVICE}, bf16 autocast, absolute error against attention in fp32")
    print("seed  matmul max  matmul mean  sdpa max  sdpa mean")
    for seed in range(5):
        e = errors(seed)
        print(f"{seed:4d}  {e['matmul'][0]:10.4f}  {e['matmul'][1]:11.5f}  {e['sdpa'][0]:8.4f}  {e['sdpa'][1]:9.5f}")


if __name__ == "__main__":
    main()
