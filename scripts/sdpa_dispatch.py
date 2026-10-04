"""Which operations F.scaled_dot_product_attention runs on MPS, from a trace.

    python scripts/sdpa_dispatch.py

The inputs are fake tensors (torch._subclasses.FakeTensorMode) that say they
live on the MPS device, so PyTorch picks its MPS code path but computes only
shapes and dtypes: nothing runs on the GPU. A dispatch mode records every aten
operation that reaches the backend, leaving out views and transposes, with the
dtype of its output. One layer's shape (32 sequences, 6 heads, 256 tokens, 64
dims), for sdpa with and without gradients and for the written-out version.
Needs a torch built with MPS support; what it prints is specific to the torch
version.
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # the repo root
from tinygpt.model import matmul_attention  # noqa: E402

SKIP = {"view", "_unsafe_view", "expand", "transpose", "t", "detach", "permute"}


class Record(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        ns, name = str(func).split(".")[:2]
        if ns == "aten" and name not in SKIP:
            outs = out if isinstance(out, (tuple, list)) else (out,)
            dtypes = ", ".join(str(o.dtype)[6:] for o in outs if isinstance(o, torch.Tensor))
            self.ops.append(f"{name} -> {dtypes}")
        return out


def trace(attention, dtype, grad, autocast):
    with FakeTensorMode():
        q, k, v = (torch.randn(32, 6, 256, 64, device="mps", dtype=dtype, requires_grad=grad) for _ in range(3))
        with torch.autocast("mps", dtype=torch.bfloat16, enabled=autocast):
            with Record() as fwd:
                y = attention(q, k, v)
        bwd = Record()
        if grad:
            dy = torch.ones_like(y)
            with bwd:
                torch.autograd.grad(y, (q, k, v), dy)
    return fwd.ops, bwd.ops


def main():
    if not torch.backends.mps.is_built():
        sys.exit("this torch has no MPS support")
    sdpa = lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)  # noqa: E731
    cases = [
        ("sdpa, bf16 under autocast, with gradients (training)", sdpa, torch.bfloat16, True, True),
        ("sdpa, bf16 under autocast, no gradients (evaluation)", sdpa, torch.bfloat16, False, True),
        ("sdpa, fp32, with gradients", sdpa, torch.float32, True, False),
        ("matmul_attention, bf16 under autocast, with gradients", matmul_attention, torch.bfloat16, True, True),
    ]
    print(f"torch {torch.__version__}, fake MPS tensors of shape 32 x 6 x 256 x 64, views left out")
    for title, attention, dtype, grad, autocast in cases:
        fwd, bwd = trace(attention, dtype, grad, autocast)
        print(f"\n{title}")
        print("  forward:  " + "\n            ".join(fwd))
        if bwd:
            print("  backward: " + "\n            ".join(bwd))


if __name__ == "__main__":
    main()
