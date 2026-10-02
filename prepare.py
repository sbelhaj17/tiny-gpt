"""Trains the tokenizer and writes the dataset as flat uint16 token files.

    python prepare.py            # data/tokenizer.json, data/train.bin, data/val.bin, data/meta.json

Each split is tokenized once. Training then reads random windows straight out
of a memory-mapped file, with no tokenizing and no Python objects per story.
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from tinygpt.bpe import EOT, Tokenizer, train_bpe

SPLITS = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}


def read_blocks(path, block_chars=1 << 24):
    """Yields pieces of a text file that each end on a story boundary."""
    with open(path, encoding="utf-8") as f:
        rest = ""
        while piece := f.read(block_chars):
            piece = rest + piece
            cut = piece.rfind(EOT)
            if cut == -1:
                rest = piece
                continue
            yield piece[:cut]
            rest = piece[cut + len(EOT):]
        if rest.strip():
            yield rest


def stories(text):
    for s in text.split(EOT):
        s = s.strip()
        if s:
            yield s


_tok = None


def _init_worker(tokenizer_path):
    global _tok
    _tok = Tokenizer.load(tokenizer_path)


def encode_block(text):
    ids = []
    nbytes = 0
    for s in stories(text):
        ids.extend(_tok.encode(s))
        ids.append(_tok.eot)
        nbytes += len(s.encode("utf-8"))
    return np.array(ids, dtype=np.uint16), nbytes


def tokenize_split(src, dst, tokenizer_path, workers):
    ntokens = nbytes = 0
    with Pool(workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool, open(dst + ".part", "wb") as out:
        # imap keeps the blocks in file order
        for ids, b in pool.imap(encode_block, read_blocks(src)):
            out.write(ids.tobytes())
            ntokens += len(ids)
            nbytes += b
    os.replace(dst + ".part", dst)
    return ntokens, nbytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--bpe_mb", type=int, default=100, help="megabytes of training text to learn merges from")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    assert args.vocab_size <= 1 << 16, "token ids are stored as uint16"

    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    t = time.perf_counter()
    sample = next(read_blocks(os.path.join(args.data_dir, SPLITS["train"]), args.bpe_mb << 20))
    tok = train_bpe(stories(sample), args.vocab_size)
    tok.save(tok_path)
    print(f"tokenizer: {tok.vocab_size} tokens from {len(sample) / 1e6:.0f} MB in {time.perf_counter() - t:.1f}s")

    meta = {"vocab_size": tok.vocab_size, "eot": tok.eot}
    for split, name in SPLITS.items():
        t = time.perf_counter()
        ntokens, nbytes = tokenize_split(
            os.path.join(args.data_dir, name), os.path.join(args.data_dir, f"{split}.bin"), tok_path, args.workers
        )
        meta[split] = {"tokens": ntokens, "bytes": nbytes}
        print(f"{split}: {ntokens:,} tokens, {nbytes / ntokens:.2f} bytes per token, {time.perf_counter() - t:.1f}s")

    with open(os.path.join(args.data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
