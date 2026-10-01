"""Compares my BPE with GPT-2's tokenizer on the validation stories.

    python scripts/compare_tokenizers.py

Needs tiktoken (pip install tiktoken) and data/ from prepare.py.
"""

import os
import sys
import time

import numpy as np
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # the repo root, for prepare and tinygpt
from prepare import stories  # noqa: E402
from tinygpt.bpe import Tokenizer  # noqa: E402


def main():
    text = open("data/TinyStoriesV2-GPT4-valid.txt", encoding="utf-8").read()
    docs = list(stories(text))
    nbytes = sum(len(s.encode("utf-8")) for s in docs)
    print(f"{len(docs):,} stories, {nbytes:,} bytes")

    for name, tok, vocab in [
        ("mine", Tokenizer.load("data/tokenizer.json"), 4096),
        ("gpt2", tiktoken.get_encoding("gpt2"), 50257),
    ]:
        encode = tok.encode if name == "mine" else tok.encode_ordinary
        t = time.time()
        ids = [encode(s) for s in docs]
        secs = time.time() - t
        n = sum(map(len, ids))
        used = len(np.unique(np.concatenate([np.asarray(x) for x in ids if x])))
        print(f"{name}: {n:,} tokens, {nbytes / n:.2f} bytes per token, "
              f"{used:,} of {vocab:,} ids used, {secs:.1f}s to encode")


if __name__ == "__main__":
    main()
