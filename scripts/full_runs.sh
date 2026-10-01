#!/bin/bash
# The training runs behind the numbers in the README.
#
#   scripts/full_runs.sh main        # the story model, about 95 minutes on an M4
#   scripts/full_runs.sh ablation    # rope against learned positions, about 2 x 30 minutes
#   scripts/full_runs.sh all         # both, one after the other
#   scripts/full_runs.sh prelim      # the short ablation I ran while building, about 3 x 9 minutes
#
# Each run checkpoints at every evaluation. If one is interrupted, run the
# same command again: a run whose checkpoint exists resumes from it, and a
# finished run does no more training. Logs go to runs/<name>/train.log, the
# full validation loss of each finished run to results/eval.jsonl.
#
# To check the script end to end in a few minutes, point it somewhere else
# and cut the runs short (a short run left in runs/ would be resumed later):
#
#   RUNS=/tmp/tg RESULTS=/tmp/tg EXTRA="--max_steps 20 --eval_interval 10 --eval_batches 2" scripts/full_runs.sh all
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
RUNS=${RUNS:-runs}
RESULTS=${RESULTS:-results}

# Keep the Mac awake while this runs (the display may still sleep).
if [ -z "$TINYGPT_AWAKE" ] && command -v caffeinate >/dev/null 2>&1; then
    TINYGPT_AWAKE=1 exec caffeinate -i "$0" "$@"
fi

# train <config> <run name> [more train.py flags]
train() {
    config=$1 name=$2
    shift 2
    mkdir -p "$RUNS/$name"
    # EXTRA is unquoted on purpose: it holds several flags
    $PY -u train.py "configs/$config.json" --out_dir "$RUNS/$name" --resume true "$@" $EXTRA 2>&1 | tee -a "$RUNS/$name/train.log"
    $PY evaluate.py "$RUNS/$name/ckpt.pt" | tee -a "$RESULTS/eval.jsonl"
}

plot() {
    out=$1
    shift
    $PY scripts/plot_runs.py "$@" --skip_first --out "$RESULTS/$out.png" | tee "$RESULTS/$out.md"
}

mkdir -p "$RESULTS"
case ${1:-all} in
main)
    train main main
    $PY sample.py "$RUNS/main/ckpt.pt" --num 5 --seed 0 > "$RESULTS/samples_main.txt"
    plot main_loss "$RUNS/main" ;;
ablation)
    train ablation_rope ablation_rope
    train ablation_learned ablation_learned
    plot ablation "$RUNS/ablation_rope" "$RUNS/ablation_learned" ;;
prelim)
    # 1000 steps each, plus the rope run again with another seed (which
    # changes both the initial weights and the order of the data), to see
    # how far apart two runs land when nothing but the seed differs.
    short="--max_steps 1000 --eval_interval 250"
    train ablation_rope prelim_rope $short
    train ablation_learned prelim_learned $short
    train ablation_rope prelim_rope_seed2 $short --seed 1338
    $PY sample.py "$RUNS/prelim_rope/ckpt.pt" --num 3 --seed 0 > "$RESULTS/samples_prelim.txt"
    plot prelim "$RUNS/prelim_rope" "$RUNS/prelim_learned" "$RUNS/prelim_rope_seed2" ;;
all)
    "$0" main
    "$0" ablation ;;
*)
    echo "usage: $0 main|ablation|prelim|all" >&2
    exit 1 ;;
esac
