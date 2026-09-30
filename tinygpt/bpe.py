"""Byte-level BPE, trained on the dataset itself.

Text is first split into chunks (words with their leading space, runs of
digits, runs of punctuation, whitespace), the same idea as GPT-2's pattern.
Merges never cross a chunk boundary, so each distinct chunk only has to be
tokenized once and the result can be cached.

Ids 0-255 are the raw bytes, then one id per merge in the order the merges
were learned, then the special end-of-text token. Any string round-trips,
because a chunk's tokens always decode back to exactly its bytes.
"""

import json
import re
from collections import Counter, defaultdict
from functools import lru_cache

# GPT-2 uses \p{L} and \p{N}, which the stdlib re lacks. [^\W\d_] is "a word
# character that is not a digit or underscore", i.e. a letter. The last two
# alternatives match any whitespace, so every character lands in some chunk.
SPLIT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+""")

EOT = "<|endoftext|>"


def merge(ids, pair, new_id):
    out = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(texts, vocab_size):
    """Learns vocab_size - 257 merges from an iterable of strings."""
    num_merges = vocab_size - 256 - 1  # 256 bytes, the merges, one special token
    assert num_merges >= 0

    counts = Counter()
    for text in texts:
        counts.update(SPLIT.findall(text))
    words = [list(w.encode("utf-8")) for w in counts]
    freqs = list(counts.values())

    # How often each adjacent pair occurs, weighted by word frequency, and which
    # words contain it. When a pair is merged only those words are revisited.
    stats = Counter()
    where = defaultdict(set)
    for i, w in enumerate(words):
        for pair in zip(w, w[1:]):
            stats[pair] += freqs[i]
            where[pair].add(i)

    merges = []
    for new_id in range(256, 256 + num_merges):
        if not stats:
            break  # every word is a single token already
        pair = max(stats, key=stats.get)
        merges.append(pair)
        # `where` can hold words that no longer contain the pair (an earlier
        # merge ate it). For those the loop below subtracts and re-adds the
        # same counts, so the stale entries cost time but not correctness.
        for i in where.pop(pair):
            w, f = words[i], freqs[i]
            for p in zip(w, w[1:]):
                stats[p] -= f
                if stats[p] == 0:
                    del stats[p]
            w = merge(w, pair, new_id)
            words[i] = w
            for p in zip(w, w[1:]):
                stats[p] += f
                where[p].add(i)
    return Tokenizer(merges)


class Tokenizer:
    def __init__(self, merges):
        self.merges = [tuple(p) for p in merges]
        self.ranks = {p: 256 + i for i, p in enumerate(self.merges)}
        self.eot = 256 + len(self.merges)
        self.vocab_size = self.eot + 1

        self.vocab = [bytes([b]) for b in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.vocab.append(EOT.encode("utf-8"))

        # TinyStories has a few tens of thousands of distinct chunks, so nearly
        # every lookup after the first few megabytes is a cache hit.
        self._encode_chunk = lru_cache(maxsize=1 << 18)(self._bpe)

    def _bpe(self, chunk):
        ids = list(chunk.encode("utf-8"))
        while len(ids) > 1:
            # Apply the earliest-learned merge available. Doing merges in the
            # order training learned them reproduces training's segmentation.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            ids = merge(ids, pair, self.ranks[pair])
        return tuple(ids)

    def encode(self, text):
        """Encodes text as ordinary characters; the EOT string is not special here."""
        out = []
        for chunk in SPLIT.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f)["merges"])
