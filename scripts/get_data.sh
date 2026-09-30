#!/bin/sh
# Fetches the GPT-4 half of TinyStories (about 2.2 GB of text) into data/.
set -e
cd "$(dirname "$0")/.."
mkdir -p data
base=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
for f in TinyStoriesV2-GPT4-valid.txt TinyStoriesV2-GPT4-train.txt; do
    if [ ! -f "data/$f" ]; then
        curl -L --fail --progress-bar -o "data/$f.part" "$base/$f"
        mv "data/$f.part" "data/$f"
    fi
done
ls -l data
