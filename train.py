"""Trains a GPT on the token files written by prepare.py.

    python train.py configs/main.json --out_dir runs/main
    python train.py configs/main.json --out_dir runs/smoke --max_steps 200
    python train.py configs/main.json --out_dir runs/main --resume true   # continue from runs/main/ckpt.pt

A config file sets any of the TrainConfig fields below; flags override it.
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import torch

from tinygpt.data import Batches
from tinygpt.model import GPT, GPTConfig

# Not time.time(): the wall clock keeps running while the laptop sleeps (the
# main run once logged four hours of sleep as training time) and jumps when the
# system adjusts it. perf_counter is monotonic and does not advance during sleep.
clock = time.perf_counter


@dataclass
class TrainConfig:
    data_dir: str = "data"
    out_dir: str = "runs/default"
    resume: bool = False
    # model
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    pos_emb: str = "rope"
    attn: str = "matmul"
    # optimization
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 5000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    # logging, evaluation, checkpoints
    log_interval: int = 20
    eval_interval: int = 500
    eval_batches: int = 40
    # speed
    device: str = "mps"
    dtype: str = "bf16"  # fp32, bf16 or fp16
    compile: bool = False
    optimizer: str = "fused"  # fused, foreach or loop
    seed: int = 1337


def parse_config(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("config", nargs="?", help="JSON file with TrainConfig fields")
    for f in fields(TrainConfig):
        kind = type(f.default)
        if kind is bool:
            kind = lambda s: s.lower() in ("1", "true", "yes")  # noqa: E731
        p.add_argument(f"--{f.name}", type=kind)
    args = vars(p.parse_args(argv))
    cfg = TrainConfig()
    if path := args.pop("config"):
        with open(path) as f:
            cfg = replace(cfg, **json.load(f))
    return replace(cfg, **{k: v for k, v in args.items() if v is not None})


def model_config(cfg, vocab_size):
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        pos_emb=cfg.pos_emb,
        attn=cfg.attn,
    )


def make_optimizer(model, cfg):
    if cfg.optimizer == "fused" and cfg.dtype == "fp16" and cfg.device.startswith("mps"):
        # In torch 2.10 the fused MPS kernel ignores the gradient scaler's
        # found-inf flag: a step whose fp16 gradients overflowed is applied
        # anyway and every weight becomes NaN. foreach skips it correctly.
        # tests/test_train.py has a check that fails once torch fixes this.
        raise ValueError("fp16 with fused AdamW is broken on MPS; use --optimizer foreach or --dtype bf16")
    # Decay the matrices (including the embedding) but not the layer norm
    # gains and biases: pulling a gain toward zero just fights the model.
    params = list(model.parameters())
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    impl = {"fused": {"fused": True}, "foreach": {"foreach": True}, "loop": {"foreach": False}}[cfg.optimizer]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), **impl)


def lr_at(step, cfg):
    """Linear warmup to cfg.lr, then a cosine down to cfg.min_lr at max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


def bits_per_byte(loss, bytes_per_token):
    """Nats per token to bits per byte of text, which does not depend on the tokenizer."""
    return loss / math.log(2) / bytes_per_token


def autocast(cfg):
    if cfg.dtype == "fp32":
        return nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[cfg.dtype]
    return torch.autocast(device_type=torch.device(cfg.device).type, dtype=dtype)


def train_step(model, opt, scaler, next_batch, cfg):
    total = 0.0
    for _ in range(cfg.grad_accum):
        x, y = next_batch()
        with autocast(cfg):
            _, loss = model(x, y)
        loss = loss / cfg.grad_accum
        # With fp16 the scaler multiplies the loss so small gradients do not
        # underflow; for fp32 and bf16 it is disabled and does nothing.
        scaler.scale(loss).backward()
        total += loss.detach()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return total


@torch.no_grad()
def evaluate(model, batches, starts, cfg):
    model.eval()
    losses = []
    for s in starts:
        x, y = batches.get(s)
        with autocast(cfg):
            _, loss = model(x, y)
        losses.append(loss)
    model.train()
    return torch.stack(losses).mean().item()


def sync(device):
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def save_checkpoint(path, **state):
    # Write then rename, so a run killed mid-save keeps its last good checkpoint.
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def main(argv=None):
    cfg = parse_config(argv)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.data_dir, "meta.json")) as f:
        meta = json.load(f)
    bytes_per_token = meta["val"]["bytes"] / meta["val"]["tokens"]

    torch.manual_seed(cfg.seed)
    mcfg = model_config(cfg, meta["vocab_size"])
    model = GPT(mcfg).to(cfg.device)
    opt = make_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(cfg.device, enabled=cfg.dtype == "fp16")

    train = Batches(os.path.join(cfg.data_dir, "train.bin"), cfg.batch_size, cfg.block_size, cfg.device, cfg.seed)
    val = Batches(os.path.join(cfg.data_dir, "val.bin"), cfg.batch_size, cfg.block_size, cfg.device)
    val_starts = val.fixed_starts(cfg.eval_batches)

    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    step, elapsed = 0, 0.0
    if cfg.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        step, elapsed = ckpt["step"], ckpt["elapsed"]
        train.rng.bit_generator.state = ckpt["data_rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        print(f"resumed from step {step}")

    # Compile a wrapper and keep `model` for state_dict, so checkpoints load
    # into a plain GPT.
    fwd = torch.compile(model) if cfg.compile else model
    tokens_per_step = cfg.batch_size * cfg.block_size * cfg.grad_accum
    print(f"{model.num_params() / 1e6:.2f}M parameters, {tokens_per_step} tokens per step")
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    log = open(os.path.join(cfg.out_dir, "log.jsonl"), "a")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()

    def checkpoint_and_eval():
        loss = evaluate(fwd, val, val_starts, cfg)
        if not math.isfinite(loss):
            # The training loss logged at this step was computed before the
            # step's update, so an update that broke the weights first shows
            # up here. Stop before it overwrites the last good checkpoint.
            raise RuntimeError(f"validation loss is {loss} at step {step}; the last good checkpoint is kept")
        bpb = bits_per_byte(loss, bytes_per_token)
        print(f"step {step:6d} | val loss {loss:.4f} | {bpb:.3f} bits per byte")
        write({"step": step, "tokens": step * tokens_per_step, "elapsed": elapsed, "val_loss": loss, "val_bpb": bpb})
        save_checkpoint(
            ckpt_path,
            model=model.state_dict(),
            optimizer=opt.state_dict(),
            scaler=scaler.state_dict(),
            model_cfg=asdict(mcfg),
            train_cfg=asdict(cfg),
            step=step,
            elapsed=elapsed,
            val_loss=loss,
            data_rng=train.rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
        )

    if step == 0:
        checkpoint_and_eval()
    sync(cfg.device)
    t_last, step_last = clock(), step
    while step < cfg.max_steps:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        loss = train_step(fwd, opt, scaler, train.next, cfg)
        step += 1

        if step % cfg.log_interval == 0 or step == cfg.max_steps:
            loss = loss.item()  # waits for the GPU, so only do it when logging
            if not math.isfinite(loss):
                # Stop before a checkpoint overwrites the last good one.
                raise RuntimeError(f"loss is {loss} at step {step}; rerun with --resume true to restart from the last checkpoint")
            now = clock()
            elapsed += now - t_last
            tok_s = (step - step_last) * tokens_per_step / (now - t_last)
            print(f"step {step:6d} | loss {loss:.4f} | lr {lr:.2e} | {tok_s:,.0f} tok/s")
            write({"step": step, "tokens": step * tokens_per_step, "elapsed": elapsed,
                   "train_loss": loss, "lr": lr, "tok_per_s": tok_s})
            t_last, step_last = now, step

        if step % cfg.eval_interval == 0 or step == cfg.max_steps:
            sync(cfg.device)
            elapsed += clock() - t_last
            checkpoint_and_eval()
            # evaluation and saving do not count as training time
            t_last, step_last = clock(), step

    log.close()


if __name__ == "__main__":
    main()
