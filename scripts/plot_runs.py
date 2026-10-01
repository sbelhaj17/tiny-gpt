"""Plots validation loss against tokens seen and prints a summary per run.

    python scripts/plot_runs.py runs/rope runs/learned --out results/ablation.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def read(run):
    with open(os.path.join(run, "log.jsonl")) as f:
        recs = [json.loads(line) for line in f]
    evals = [r for r in recs if "val_loss" in r]
    speeds = [r["tok_per_s"] for r in recs if "tok_per_s" in r]
    return evals, speeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="results/loss.png")
    p.add_argument("--skip_first", action="store_true", help="leave out the step-0 evaluation, which squashes the y axis")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    print("| run | steps | tokens | training time | val loss | bits per byte | median tok/s |")
    print("|---|---|---|---|---|---|---|")
    for run, color in zip(args.runs, COLORS):
        evals, speeds = read(run)
        if args.skip_first:
            evals = [e for e in evals if e["step"] > 0]
        name = os.path.basename(os.path.normpath(run))
        xs = [e["tokens"] / 1e6 for e in evals]
        ys = [e["val_loss"] for e in evals]
        ax.plot(xs, ys, color=color, lw=2, solid_capstyle="round", label=name)
        last = evals[-1]
        speed = sorted(speeds)[len(speeds) // 2] if speeds else float("nan")
        print(f"| {name} | {last['step']} | {last['tokens'] / 1e6:.1f}M | {last['elapsed'] / 60:.1f} min "
              f"| {last['val_loss']:.3f} | {last['val_bpb']:.3f} | {speed:,.0f} |")

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
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
