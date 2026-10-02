#!/bin/bash
# The training runs behind the numbers in the README.
#
#   scripts/full_runs.sh main           # the story model, about 95 minutes on an M4
#   scripts/full_runs.sh ablation       # rope against learned positions, about 2 x 30 minutes
#   scripts/full_runs.sh ablation_seed  # the rope run again with another seed, about 30 minutes
#   scripts/full_runs.sh ablation_timed # learned positions given rope's time, about 27 minutes
#   scripts/full_runs.sh all            # main, ablation and ablation_seed, one after the other
#   scripts/full_runs.sh short          # the main model for 1000 steps, about 9 minutes
#
# Keep the Mac on AC power with the lid open. caffeinate below stops idle
# sleep, but closing the lid still puts the machine to sleep, and a run then
# only advances during the brief wakes macOS makes while asleep.
#
# Each run checkpoints at every evaluation. If one is interrupted, run the
# same command again: a run whose checkpoint exists resumes from it, and a
# finished run does no more training. Logs go to runs/<name>/train.log, the
# full validation loss of each finished run to results/eval.jsonl (once per
# checkpoint and step, however often the script is rerun).
#
# To check the script end to end, point it somewhere else and cut the runs
# short (a short run left in runs/ would be resumed by the real one later):
#
#   RUNS=/tmp/tg RESULTS=/tmp/tg EXTRA="--max_steps 20 --eval_interval 10 --eval_batches 2" scripts/full_runs.sh ablation
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
RUNS=${RUNS:-runs}
RESULTS=${RESULTS:-results}

# Keep the Mac from idle-sleeping while this runs (see the note on the lid above).
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
    $PY evaluate.py "$RUNS/$name/ckpt.pt" --append "$RESULTS/eval.jsonl"
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
ablation_seed)
    # A new seed changes both the initial weights and the order of the data.
    # How far this lands from ablation_rope is the noise a difference between
    # rope and learned has to beat.
    train ablation_rope ablation_rope_seed2 --seed 1338
    plot ablation_seed "$RUNS/ablation_rope" "$RUNS/ablation_learned" "$RUNS/ablation_rope_seed2" ;;
ablation_timed)
    # The learned-position run is about 16% faster per step, so equal tokens
    # is not equal time. This gives it as many steps as it can take in the
    # time ablation_rope took: 4000 x 24,630 / 21,283 tok/s = 4,629.
    train ablation_learned ablation_learned_timed --max_steps 4629
    plot ablation_timed "$RUNS/ablation_rope" "$RUNS/ablation_rope_seed2" "$RUNS/ablation_learned" "$RUNS/ablation_learned_timed" ;;
short)
    # The run I used to check the whole pipeline while building.
    train main short --max_steps 1000 --eval_interval 250
    $PY sample.py "$RUNS/short/ckpt.pt" --num 3 --seed 0 > "$RESULTS/samples_short.txt"
    plot short_loss "$RUNS/short" ;;
all)
    "$0" main
    "$0" ablation
    "$0" ablation_seed ;;
*)
    echo "usage: $0 main|ablation|ablation_seed|ablation_timed|all|short" >&2
    exit 1 ;;
esac
