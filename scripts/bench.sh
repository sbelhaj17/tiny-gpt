#!/bin/sh
# Times training steps under different settings and appends one JSON line per
# setting to results/bench.jsonl. Each setting runs in its own process, so
# compiled kernels and cached GPU memory do not carry over between them.
#
#   scripts/bench.sh                  # every group, about 15 minutes on an M4
#   scripts/bench.sh precision batch  # only some groups
#
# REPEATS=2 runs the whole list twice, so slow drift (heat, other programs)
# shows up as disagreement between passes. BEFORE is a command run before
# each setting, e.g. one that waits until nothing else is using the GPU.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
OUT=${OUT:-results/bench.jsonl}
REPEATS=${REPEATS:-1}
mkdir -p "$(dirname "$OUT")"

run() {
    ${BEFORE:-true}
    $PY bench.py "$@" | tail -n 1 | tee -a "$OUT"
}

group() {
    case $1 in
    precision)
        run --dtype fp32
        run --dtype fp16
        run --dtype bf16 ;;
    optimizer)
        run --optimizer fused
        run --optimizer foreach
        run --optimizer loop ;;
    attention)
        run --attn sdpa
        run --attn naive ;;
    compile)
        run --compile false
        run --compile true ;;
    data)
        run
        run --same_batch ;;
    batch)
        for b in 8 16 32 64 128; do run --batch_size $b; done ;;
    vocab)
        # the same model with GPT-2's vocabulary instead of my 4096-token BPE
        run --vocab_size 4096
        run --vocab_size 50257 ;;
    size)
        run --n_layer 4 --n_head 4 --n_embd 256
        run --n_layer 6 --n_head 6 --n_embd 384
        run --n_layer 8 --n_head 8 --n_embd 512 ;;
    *)
        echo "unknown group $1" >&2
        exit 1 ;;
    esac
}

groups=${*:-precision optimizer attention compile data batch vocab size}
i=0
while [ $i -lt "$REPEATS" ]; do
    for g in $groups; do group "$g"; done
    i=$((i + 1))
done
