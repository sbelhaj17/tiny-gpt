import numpy as np
import torch


class Batches:
    """Random windows from a flat file of uint16 token ids.

    Stories are packed back to back with an end-of-text token between them, so
    a window can start mid-story and cross into the next one. The model learns
    to ignore what came before an end-of-text token.
    """

    def __init__(self, path, batch_size, block_size, device, seed=0):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.rng = np.random.default_rng(seed)

    def random_starts(self, rng=None):
        rng = rng or self.rng
        # A window needs block_size + 1 tokens (inputs plus the shifted
        # targets), so the last start is len - block_size - 1. integers()
        # excludes its upper bound.
        return rng.integers(0, len(self.tokens) - self.block_size, size=self.batch_size)

    def get(self, starts):
        T = self.block_size
        w = np.stack([self.tokens[s : s + T + 1] for s in starts]).astype(np.int64)
        w = torch.from_numpy(w).to(self.device)
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()

    def next(self):
        return self.get(self.random_starts())

    def fixed_starts(self, num_batches, seed=0):
        """The same windows every time, so validation losses are comparable
        across steps and across runs with the same batch and block size."""
        rng = np.random.default_rng(seed)
        return [self.random_starts(rng) for _ in range(num_batches)]
