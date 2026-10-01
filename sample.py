"""Writes stories with a trained checkpoint.

    python sample.py runs/main/ckpt.pt
    python sample.py runs/main/ckpt.pt --prompt "Once upon a time, a little fox" --num 3 --temperature 0.8
"""

import argparse
import os

import torch

from tinygpt.bpe import Tokenizer
from tinygpt.model import GPT, GPTConfig


def load(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.load(os.path.join(ckpt["train_cfg"]["data_dir"], "tokenizer.json"))
    return model.to(device).eval(), tok, ckpt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--prompt", default="")
    p.add_argument("--num", type=int, default=3)
    p.add_argument("--max_tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()

    model, tok, ckpt = load(args.ckpt, args.device)
    print(f"step {ckpt['step']}, val loss {ckpt['val_loss']:.3f}\n")
    torch.manual_seed(args.seed)
    # In training every story follows an end-of-text token, so starting from
    # one tells the model it is at the beginning of a story.
    start = [tok.eot] + tok.encode(args.prompt)
    idx = torch.tensor([start] * args.num, device=args.device)
    out = model.generate(idx, args.max_tokens, args.temperature, args.top_k, stop_token=tok.eot)
    for row in out.tolist():
        story = row[1:]
        if tok.eot in story:
            story = story[: story.index(tok.eot)]
        print(tok.decode(story).strip())
        print("\n---\n")


if __name__ == "__main__":
    main()
