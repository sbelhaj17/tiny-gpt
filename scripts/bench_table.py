"""Turns results/bench.jsonl into a markdown table, one row per setting.

    python scripts/bench_table.py [results/bench.jsonl]

A setting that was timed more than once (scripts/bench.sh repeats the
baseline in several groups, and REPEATS=2 repeats everything) gets the median
step time and the spread, so a difference can be compared with the noise.
"""

import json
import os
import statistics
import sys
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # the repo root, for train
from train import TrainConfig  # noqa: E402

KEYS = ["n_layer", "n_embd", "batch_size", "dtype", "attn", "optimizer", "compile", "vocab_size", "same_batch"]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/bench.jsonl"
    with open(path) as f:
        recs = [json.loads(line) for line in f]
    base = asdict(TrainConfig()) | {"vocab_size": 4096, "same_batch": False}

    rows = {}
    for r in recs:
        key = tuple(r[k] for k in KEYS)
        rows.setdefault(key, []).append(r)

    print("| setting | params | ms per step | spread | tokens/s | GPU memory |")
    print("|---|---|---|---|---|---|")
    for key, rs in rows.items():
        diff = [f"{k}={v}" for k, v in zip(KEYS, key) if v != base[k]]
        name = ", ".join(diff) or "baseline"
        ms = [r["ms_per_step"] for r in rs]
        med = statistics.median(ms)
        spread = f"{min(ms):.0f}-{max(ms):.0f} ({len(ms)} runs)" if len(ms) > 1 else "1 run"
        tok_s = statistics.median(r["tok_per_s"] for r in rs)
        print(f"| {name} | {rs[0]['params_m']}M | {med:.0f} | {spread} | {tok_s:,.0f} | {rs[0].get('mem_gb', '')} GB |")


if __name__ == "__main__":
    main()
