"""Times full training steps (forward, backward, clip, AdamW) on real batches.

    python bench.py --dtype fp32 --batch_size 64
    python bench.py --vocab_size 50257    # as if using GPT-2's tokenizer

Takes the same flags as train.py (defaults: TrainConfig, the 6-layer model)
plus the ones below, and prints one JSON line.
Each configuration should run in its own process, so one run's compiled
kernels and cached memory do not leak into the next (scripts/bench.sh does this).
"""

import argparse
import json
import os
import time
from dataclasses import asdict

import torch

from tinygpt.data import Batches
from tinygpt.model import GPT
from train import make_optimizer, model_config, parse_config, sync, train_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--same_batch", action="store_true", help="reuse one batch, to see what loading data costs")
    args, rest = p.parse_known_args()
    cfg = parse_config(rest)

    torch.manual_seed(0)
    model = GPT(model_config(cfg, args.vocab_size)).to(cfg.device)
    opt = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.dtype == "fp16")
    fwd = torch.compile(model) if cfg.compile else model
    batches = Batches(os.path.join(cfg.data_dir, "train.bin"), cfg.batch_size, cfg.block_size, cfg.device)
    next_batch = batches.next
    if args.same_batch:
        fixed = batches.next()
        next_batch = lambda: fixed  # noqa: E731

    t = time.perf_counter()
    for _ in range(args.warmup):
        train_step(fwd, opt, scaler, next_batch, cfg)
    sync(cfg.device)
    warmup_s = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(args.steps):
        loss = train_step(fwd, opt, scaler, next_batch, cfg)
    loss.item()
    sync(cfg.device)
    dt = time.perf_counter() - t

    tokens = args.steps * cfg.batch_size * cfg.block_size * cfg.grad_accum
    keys = ["n_layer", "n_embd", "block_size", "batch_size", "grad_accum", "pos_emb", "attn", "dtype", "compile", "optimizer"]
    rec = {k: v for k, v in asdict(cfg).items() if k in keys}
    rec.update(
        vocab_size=args.vocab_size,
        same_batch=args.same_batch,
        params_m=round(model.num_params() / 1e6, 2),
        ms_per_step=round(1000 * dt / args.steps, 1),
        tok_per_s=round(tokens / dt),
        warmup_s=round(warmup_s, 1),
        loss=round(loss.item(), 3),
    )
    if cfg.device == "mps":
        rec["mem_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
