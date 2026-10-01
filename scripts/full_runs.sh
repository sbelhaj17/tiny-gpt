#!/bin/bash
# The long runs behind the final numbers in the README.
#
#   scripts/full_runs.sh main        # the story model, about 95 minutes on an M4
#   scripts/full_runs.sh ablation    # rope against learned positions, about 2 x 32 minutes
#   scripts/full_runs.sh all         # both, one after the other
#
# Each run checkpoints at every evaluation. If one is interrupted, run the
# same command again: a run whose checkpoint exists resumes from it, and a
# finished run does no more training. Logs go to runs/<name>/train.log.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

# Keep the Mac awake while this runs (the display may still sleep).
if [ -z "$TINYGPT_AWAKE" ] && command -v caffeinate >/dev/null 2>&1; then
    TINYGPT_AWAKE=1 exec caffeinate -i "$0" "$@"
fi

train() {
    name=$1
    mkdir -p "runs/$name"
    $PY -u train.py "configs/$name.json" --out_dir "runs/$name" --resume true 2>&1 | tee -a "runs/$name/train.log"
    $PY evaluate.py "runs/$name/ckpt.pt" | tee -a results/final_eval.jsonl
}

mkdir -p results
case ${1:-all} in
main)
    train main
    $PY sample.py runs/main/ckpt.pt --num 5 --seed 0 > results/samples_main.txt
    $PY scripts/plot_runs.py runs/main --skip_first --out results/main_loss.png ;;
ablation)
    train ablation_rope
    train ablation_learned
    $PY scripts/plot_runs.py runs/ablation_rope runs/ablation_learned --skip_first --out results/ablation.png ;;
all)
    "$0" main
    "$0" ablation ;;
*)
    echo "usage: $0 main|ablation|all" >&2
    exit 1 ;;
esac
