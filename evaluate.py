"""Loss over the whole validation split, for final numbers.

    python evaluate.py runs/main/ckpt.pt
    python evaluate.py runs/main/ckpt.pt --append results/eval.jsonl

Training only evaluates a fixed sample of windows, to keep evaluation cheap.
This cuts all of val.bin into back-to-back windows of the model's block size
and averages the loss over every token, so it has no sampling noise. The
windows do not overlap, so a token is predicted from anywhere between 0 and
block_size - 1 tokens of context, about half a block on average, as in training.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from sample import load
from train import bits_per_byte


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


def already_evaluated(path, ckpt_path, step):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["ckpt"] == ckpt_path and rec["step"] == step:
                return rec
    return None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--append", help="JSONL file to add the result to; a checkpoint already in it at the same step is not evaluated again")
    args = p.parse_args(argv)

    model, _, ckpt = load(args.ckpt, args.device)
    # Rerunning a finished run's script should not add a second line for the
    # same checkpoint.
    if args.append and (rec := already_evaluated(args.append, args.ckpt, ckpt["step"])):
        print(json.dumps(rec))
        return rec

    data_dir = ckpt["train_cfg"]["data_dir"]
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    tokens = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    t = time.perf_counter()
    loss, count = full_loss(model, tokens, model.cfg.block_size, args.batch_size, args.device)
    bpb = bits_per_byte(loss, meta["val"]["bytes"] / meta["val"]["tokens"])
    rec = {"ckpt": args.ckpt, "step": ckpt["step"], "tokens": count, "val_loss": round(loss, 4),
           "val_bpb": round(bpb, 4), "seconds": round(time.perf_counter() - t, 1)}
    print(json.dumps(rec))
    if args.append:
        with open(args.append, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    main()
