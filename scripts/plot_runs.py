"""Plots validation loss against tokens seen and prints a summary per run.

    python scripts/plot_runs.py runs/rope runs/learned --out results/ablation.png
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def read(run):
    """The run's evaluation and training records, in step order.

    A run killed between checkpoints and resumed logs the steps after its last
    checkpoint twice; the later record of each step is the one that counts.
    """
    with open(os.path.join(run, "log.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    evals = {r["step"]: r for r in recs if "val_loss" in r}
    trains = {r["step"]: r for r in recs if "tok_per_s" in r}
    return [evals[s] for s in sorted(evals)], [trains[s] for s in sorted(trains)]


def eval_batches(run):
    with open(os.path.join(run, "config.json")) as f:
        return json.load(f)["eval_batches"]


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="results/loss.png")
    p.add_argument("--skip_first", action="store_true", help="leave out the step-0 evaluation, which squashes the y axis")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    # "logged time" is the training time train.py measured. "at median speed"
    # is the tokens divided by the median speed, which a stall in the middle of
    # a run (the laptop sleeping, another program on the GPU) does not move;
    # for a run without stalls the two agree. The loss columns are the last
    # evaluation train.py logged, on its fixed validation batches, not
    # evaluate.py's pass over the whole split (results/eval.jsonl).
    n = "/".join(str(b) for b in sorted({eval_batches(run) for run in args.runs}))
    print(f"| run | steps | tokens | logged time | at median speed | val loss ({n} batches) "
          f"| bits per byte ({n} batches) | median tok/s |")
    print("|---|---|---|---|---|---|---|---|")
    for run, color in zip(args.runs, COLORS):
        evals, trains = read(run)
        if args.skip_first:
            evals = [e for e in evals if e["step"] > 0]
        name = os.path.basename(os.path.normpath(run))
        xs = [e["tokens"] / 1e6 for e in evals]
        ys = [e["val_loss"] for e in evals]
        ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", label=name)
        last = evals[-1]
        speed = median(r["tok_per_s"] for r in trains)
        print(f"| {name} | {last['step']} | {last['tokens'] / 1e6:.1f}M | {last['elapsed'] / 60:.1f} min "
              f"| {last['tokens'] / speed / 60:.1f} min | {last['val_loss']:.3f} | {last['val_bpb']:.3f} | {speed:,.0f} |")

    ax.set_xlabel("training tokens (millions)", color=INK)
    ax.set_ylabel("validation loss (nats per token)", color=INK)
    ax.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    if len(args.runs) > 1:
        ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
