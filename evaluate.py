"""Loss over the whole validation split, for final numbers.

    python evaluate.py runs/main/ckpt.pt

Training only evaluates a fixed sample of windows, to keep evaluation cheap.
This cuts all of val.bin into back-to-back windows of the model's block size
and averages the loss over every token, so it has no sampling noise.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from sample import load


@torch.no_grad()
def full_loss(model, tokens, block_size, batch_size, device):
    n = (len(tokens) - 1) // block_size  # windows; the leftover tail is dropped
    total, count = 0.0, 0
    for i in range(0, n, batch_size):
        starts = range(i * block_size, min(n, i + batch_size) * block_size, block_size)
        w = torch.from_numpy(np.stack([tokens[s : s + block_size + 1] for s in starts]).astype(np.int64)).to(device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            _, loss = model(w[:, :-1], w[:, 1:])
        total += loss.item() * w[:, 1:].numel()
        count += w[:, 1:].numel()
    return total / count, count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    model, _, ckpt = load(args.ckpt, args.device)
    data_dir = ckpt["train_cfg"]["data_dir"]
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    tokens = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    t = time.time()
    loss, count = full_loss(model, tokens, model.cfg.block_size, args.batch_size, args.device)
    bpb = loss / math.log(2) / (meta["val"]["bytes"] / meta["val"]["tokens"])
    print(json.dumps({"ckpt": args.ckpt, "step": ckpt["step"], "tokens": count, "val_loss": round(loss, 4),
                      "val_bpb": round(bpb, 4), "seconds": round(time.time() - t, 1)}))


if __name__ == "__main__":
    main()
