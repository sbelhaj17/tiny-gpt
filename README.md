# tiny-gpt

A small GPT written from scratch in PyTorch and trained on my laptop's GPU to write children's stories.

Everything except PyTorch itself is here: a byte-level BPE tokenizer, a data pipeline that tokenizes the dataset once into a flat file of `uint16` ids, a decoder-only transformer, the training loop, and scripts for sampling, evaluation and timing. There is no Hugging Face code. The dataset is TinyStories, about 2.7 million short stories written by GPT-4 in the vocabulary of a small child. Ronen Eldan and Yuanzhi Li made it to show that models with a few million parameters can learn to write coherent English if the language is simple enough, which makes it the right size of problem for a laptop.

I built it to understand each part of a GPT well enough to write it myself, and to find out what actually makes training fast on the hardware I have rather than on the GPUs most guides assume. Several of the usual answers turned out not to apply.

All numbers below were measured on an Apple M4 MacBook Pro (10 CPU cores, 10 GPU cores, 24 GB of memory), PyTorch 2.10 on the MPS backend, Python 3.14. The timings and the short run were on battery with Low Power Mode off; the full runs were on AC power once they ran at full speed (see below).

## How it works

**Tokenizer** (`tinygpt/bpe.py`). Byte-level BPE, the same algorithm as GPT-2's. Text is split into chunks with a GPT-2-style pattern (words with their leading space, numbers, punctuation, whitespace), merges never cross a chunk boundary, and ids 0-255 are raw bytes, so any string round-trips. I learn 3,839 merges from the first 100 MB of the training text, which with the 256 bytes and an end-of-text token makes a 4,096-token vocabulary. Encoding caches the tokens for each distinct chunk; TinyStories has few distinct words, so almost every lookup is a hit.

I wrote my own instead of using GPT-2's tokenizer through `tiktoken` because GPT-2's vocabulary is far too big for this model and this text. On the 27,630 validation stories (`scripts/compare_tokenizers.py`):

| | vocabulary | tokens | bytes per token | ids actually used |
|---|---|---|---|---|
| my BPE | 4,096 | 5,550,785 | 3.98 | 3,809 |
| GPT-2 | 50,257 | 5,441,394 | 4.06 | 11,515 |

GPT-2's tokenizer makes only 2% fewer tokens, and three quarters of its vocabulary never appears. With the model below, its 50,257 x 384 embedding table would have 19.3M parameters, more than the 10.6M in all the transformer layers together, and the output layer that scores every token at every position would dominate the step: a training step took 2.6 times as long (1,093 ms against 420 ms) and 11.3 GB of GPU memory instead of 3.5 GB.

**Data** (`prepare.py`, `tinygpt/data.py`). `prepare.py` trains the tokenizer, then tokenizes the training and validation files once, in parallel, into `train.bin` (552,563,524 tokens) and `val.bin` (5,578,415 tokens), with an end-of-text token after each story. Training memory-maps the file and cuts random 256-token windows out of it, so loading a batch is 32 slices of an array. Feeding the same batch every step instead of fresh ones made no difference to step time (422 ms against 420 ms), so data loading costs nothing measurable.

**Model** (`tinygpt/model.py`). A pre-LayerNorm decoder-only transformer: token embedding, 6 blocks of causal self-attention and a GELU MLP, a final LayerNorm, and an output layer that shares its weights with the token embedding. Positions are either rotary embeddings applied to the queries and keys (the default) or a learned position embedding added to the input; that choice is the ablation below. Linear layers have no biases, and the output projections of each block start smaller, scaled by 1/sqrt(2 x layers), so the residual stream does not grow with depth at initialization. 384 dimensions, 6 heads of 64, 256-token context: 12.2M parameters, 1.6M of them in the embedding. A story averages about 200 tokens, so 256 tokens of context holds most of one.

**Training** (`train.py`). AdamW with betas 0.9 and 0.95 and weight decay 0.1 on the weight matrices only, a 200-step linear warmup to 1e-3 and a cosine decay to 1e-4, gradient clipping at 1.0, and bf16 autocast with the loss computed in fp32. Every 500 steps it measures validation loss on the same 40 fixed batches, so evaluations are comparable across steps and across runs, and writes a checkpoint (to a temporary file and then renamed, so a run killed mid-save keeps its last good one, and not at all if the validation loss is not finite). A checkpoint holds the model, the optimizer, the sampler's random state and the step. A test kills a run right after a checkpoint, partway through the warmup and cosine, resumes it, and checks that it ends with exactly the weights of a run that never stopped; that test runs on the CPU in fp32, because GPU kernels are not promised to be bitwise deterministic. `evaluate.py` gives the final number, the loss over every token of the validation split (about 100 seconds for 5.58M tokens), and also reports it in bits per byte of text, which does not depend on the tokenizer. It cuts the split into back-to-back 256-token windows that do not overlap, so a token is predicted from between 0 and 255 tokens of context, about 128 on average, the same as in training. A sliding window that gave every token the full context would score lower; I have not run one.

**Sampling** (`sample.py`). Starts from an end-of-text token, as every story in training does, and samples with temperature and optional top-k until each story has ended.

## Speed on the M4

`scripts/bench.sh` times 30 full training steps (forward, backward, clipping, AdamW) on real batches after 10 warmup steps, each setting in its own process, and changes one thing at a time from the baseline: bf16 autocast, attention written out as matmuls, fused AdamW, batch of 32 x 256 tokens. I ran the whole sweep twice. The baseline appears in several groups, so it was timed 16 times; the step time column is the median.

| setting | parameters | ms per step | range | tokens/s | GPU memory |
|---|---|---|---|---|---|
| baseline | 12.2M | 420 | 408-485 (16 runs) | 19,494 | 3.5 GB |
| fp32, no autocast | 12.2M | 432 | 418-438 (4) | 18,960 | 3.5 GB |
| fp16 autocast (foreach AdamW, see below) | 12.2M | 429 | 423-435 (2) | 19,107 | 3.5 GB |
| PyTorch's fused attention (`sdpa`) | 12.2M | 459 | 454-464 (2) | 17,838 | 3.5 GB |
| `sdpa`, fp32 | 12.2M | 443 | 441-445 (2) | 18,499 | 3.5 GB |
| `torch.compile` | 12.2M | 512 | 508-515 (2) | 16,008 | 2.3 GB |
| foreach AdamW | 12.2M | 419 | 416-432 (4) | 19,548 | 3.5 GB |
| for-loop AdamW | 12.2M | 425 | 422-428 (2) | 19,276 | 3.5 GB |
| batch 8 | 12.2M | 109 | 108-110 (2) | 18,786 | 1.3 GB |
| batch 16 | 12.2M | 215 | 211-218 (2) | 19,092 | 2.4 GB |
| batch 64 | 12.2M | 903 | 843-963 (2) | 18,225 | 4.7 GB |
| batch 128 | 12.2M | 1,801 | 1,686-1,916 (2) | 18,272 | 8.2 GB |
| GPT-2's vocabulary | 29.9M | 1,093 | 1,044-1,141 (2) | 7,512 | 11.3 GB |
| 4 layers, 256 dims | 4.2M | 181 | 177-185 (2) | 45,232 | 2.4 GB |
| 8 layers, 512 dims | 27.3M | 808 | 796-820 (2) | 10,135 | 4.6 GB |

The slow tail of the baseline (468-485 ms) and the high ends of the batch 64, batch 128 and GPT-2 rows all come from the same few minutes near the end of the second pass, when another program started using the GPU. Leaving those out, the baseline ran between 408 and 430 ms, so differences under about 5% are noise. The raw lines are in `results/bench.jsonl`, and `results/bench_initial.jsonl` is the first sweep, before I changed the attention default.

What helped, what did not, and why:

- **Half precision barely helps.** On this GPU an fp16 or bf16 matrix multiply runs at about 3.7 TFLOP/s against 3.2-3.3 in fp32, only 13-16% faster (`scripts/profile_step.py`). My guess is that the casts autocast inserts eat most of that; I did not time them separately. bf16 and fp32 came out within about 4% of each other in both directions, which is close to the noise. I kept bf16 because it is not slower, and on a GPU with real half-precision throughput it would matter.
- **Writing attention out by hand beat the fused kernel.** At this size, `F.scaled_dot_product_attention` on MPS took 19.0 ms for a forward and backward pass of one layer in bf16; two matmuls, a mask and a softmax took 9.3 ms. In the whole model the switch took a step from 459 to 420 ms, 9% faster, so the matmul version is now the default. The tests check that the two give the same logits in fp32 on the CPU. They are not quite the same computation under bf16: the written-out version computes the attention scores in bf16 (autocast runs only the softmax in fp32), so it is less precise. Against attention in fp32, on five sets of random inputs, its mean error was 0.00061 against 0.00044 for the fused kernel, and its largest error 0.016-0.020 against 0.012-0.015 (`scripts/attention_precision.py`, output in `results/attention_precision.txt`). I have not compared training runs with the two; the runs with the matmul version trained without trouble.
- **`torch.compile` works on MPS but made steps 22% slower.** It did cut GPU memory by a third, but memory was never the limit here.
- **The optimizer implementation does not matter.** Clipping and AdamW together take 7 ms of a 420 ms step, so fused, foreach and a plain loop are within noise of each other.
- **Batch size does not change throughput.** It is flat from 8 to 128 sequences (18,200 to 19,500 tokens/s), so even 8 sequences of 256 tokens keep the GPU busy, and a bigger batch only means fewer optimizer steps for the same tokens.
- **What does matter is model size and vocabulary.** Those are the rows that move by factors, not percent.
- **Rotary embeddings are not free.** After the sweep I timed the two position encodings (`scripts/bench.sh position`): 422 ms a step with rotary against 344 ms with learned positions. Rotating the queries and keys is little arithmetic but several small elementwise kernels per layer, forward and backward, and that kind of work is what this GPU does worst. Part of it was my own doing: the cos and sin tables are fp32, so under autocast the rotation ran in fp32 and was cast back. Casting the tables to the activations' dtype instead brought rotary to 400 ms (`results/bench_position_before.jsonl` and `results/bench_position.jsonl`), still 16% slower than learned positions. That is one run on each side, and a 5% gain is at the edge of what I counted as noise above, so I would not lean on its size; the gap to learned positions showed up in both pairs of runs.

The table and the step breakdown below were measured before that change, so the baseline is probably about 400 ms now rather than 420.

Where a step goes (`scripts/profile_step.py`, output in `results/profile.txt`): the forward pass takes about 146 ms, the backward 270 ms, and clipping plus AdamW 7 ms. The model needs about 0.66 TFLOP per step, which at the measured bf16 matmul rate would take about 180 ms, so more than half the step goes to everything else: layer norms, GELU, softmax, the cross-entropy over 4,096 classes, residual adds. Those are small operations, each its own kernel, and most likely limited by memory bandwidth rather than arithmetic; that is an inference from the FLOP count, since I did not profile them one by one. They are what a compiler is supposed to fuse, and on MPS `torch.compile` did not make that pay off.

### fp16 and fused AdamW turn the weights into NaN

My first fp16 timing ended with a loss of NaN. In fp16, gradients can overflow, so a gradient scaler multiplies the loss up and skips any step whose gradients came out infinite. The fused AdamW kernel on MPS ignores that signal: given an infinite gradient, it applies the step anyway and every weight becomes NaN. Its step counter is rolled back to 0 afterwards, so the kernel does receive the scaler's flag; it just does not stop the update. The foreach implementation skips the step as it should, and on the CPU both do. I did not find this in the PyTorch issue tracker and have not reported it yet. `make_optimizer` now refuses fp16 with fused AdamW on MPS, and `tests/test_train.py` has a test that reproduces the bug in a few lines; it will start failing when PyTorch fixes it, which is the signal to drop the check.

## Training results

### A seven-minute run

To check the whole pipeline I trained the main model for 1,000 steps instead of 13,000 (`scripts/full_runs.sh short`: the same settings, with the cosine squeezed into 1,000 steps). That is 8.2M tokens and 7.2 minutes of training at a median 18,700 tokens/s, a little under the table because another program was using the GPU at the same time and the rotary change above had not been made yet.

The loss over the whole validation split came out at 2.139 nats per token, or 0.780 bits per byte. For scale, a model that knew nothing would score ln 4096 = 8.32; this one started at 8.41. The curve is in `results/short_loss.png`. The first of three samples at temperature 0.8 (`results/samples_short.txt`) begins:

> Once there was a little girl named Lily. She loved to play outside in the sun. One day, Lily found a big orange ball that she wanted to go inside. She walked down to the ball and asked, "Can I play with your ball?"

and a few sentences later Lily "quickly turned into a seal on its back". After seven minutes the grammar, the names and the shape of a TinyStories story are there and the plot is not. The main run sees 13 times as many tokens.

### The full runs

```
scripts/full_runs.sh main           # about 95 minutes
scripts/full_runs.sh ablation       # about 55 minutes
scripts/full_runs.sh ablation_seed  # about 30 minutes, after ablation
```

- `configs/main.json`: the model above for 13,000 steps of 8,192 tokens, 106.5M tokens, about a fifth of the training split. At 400 ms a step that is 87 minutes of training, plus about three minutes of periodic evaluation and checkpoints and two for the final pass over the validation split.
- `configs/ablation_rope.json` and `configs/ablation_learned.json`: the same model with rotary and with learned positions, 4,000 steps (32.8M tokens) each, with the same seed, so they see the same data in the same order. Only the position encoding differs, though the initial weights do too, because the learned run draws its position table from the same random stream. The budget is equal in tokens, not in time: the learned run should take about 23 minutes of training and the rotary run about 27. If rotary wins by less than what 16% more steps would buy, learned positions are the better choice on this machine.
- `ablation_seed` repeats the rotary run with a different seed, which changes the initial weights and the order of the data. The gap between the two rotary runs is the noise that a difference between rotary and learned has to beat.

Each run writes `runs/<name>/log.jsonl` and checkpoints; the script then adds the full validation loss to `results/eval.jsonl` and writes loss curves and a summary table to `results/`, and five sample stories from the main model to `results/samples_main.txt`.

**Why 12M parameters.** The Chinchilla rule of thumb is about 20 training tokens per parameter, with training compute C = 6ND for N parameters and D tokens. Putting D = 20N gives N = sqrt(C / 120). Counted the same way, the 12.2M model at 400 ms a step runs at 1.5 TFLOP/s, so 90 minutes is C = 8.1e15 FLOP and N comes out at 8.2M, trained on 164M tokens. That assumes a smaller model uses the GPU as well, and the table says it does not: by the same count the 4.2M model ran at 1.1 TFLOP/s against 1.4 for the 12.2M one (both timed before the rotary change), so the real optimum is somewhat below 8M. The main run gives its 12.2M parameters about 9 tokens each, so by this rule the model is a little too big for the budget, not too small. That is why I did not go bigger: the 27M model in the table would see about 55M tokens in the same time, 2 per parameter.

**Validation and test data.** TinyStories ships only a training and a validation split, so there is no separate test set. The validation split is used for the evaluations during training, for the ablation and for the final numbers. I did not tune anything on it: the main run's settings were fixed before it started, the short run used the same ones, and the ablation does not feed back into the main run.

**The first launch slept through most of the night.** A watcher script I had left to start the full runs once the rest of the pipeline finished launched them at 19:59, while the laptop was asleep with its lid closed, on battery. The `caffeinate -i` in `full_runs.sh` stops idle sleep, not lid-closed sleep. For the next 4.4 hours the main run moved only during the short wakes macOS makes while asleep, about 300 steps in all, and then ran at full speed once the lid was opened at 00:41 (`pmset -g log` has the sleep and wake times). `train.py` timed itself with `time.time()`, the wall clock, which kept counting through the sleep: at step 300 its log claimed 4.4 hours of training for about two minutes of work. Every timer now uses `time.perf_counter()`, which is monotonic and on macOS does not advance during sleep, and `full_runs.sh` says to keep the lid open and the power connected. The main run was already going with the old timer, so its logged time is wrong. `scripts/plot_runs.py` now prints two times for each run: the logged one, and the tokens divided by the median speed, which a stall does not move. For the main run only the second means anything; for the runs started after the fix the two should agree.

The ablation runs also started after I fixed an off-by-one in the window sampler (the last window in the file could never be drawn). Their 40 fixed validation batches are therefore not the same windows as the main run's, so their curves compare with each other but not with the main run's. The full-split numbers from `evaluate.py` do not depend on the sampler.

**Status.** The main run was at step 3,850 of 13,000 at 01:06, at about 21,000 tokens/s, with the ablation queued behind it. Its results are not in this README yet.

## Testing

```
.venv/bin/python -m pytest -q
```

49 tests, about two seconds (two of them need the Apple GPU and are skipped elsewhere). The ones I would point to:

- **Causal masking.** Changing tokens at position t and later must leave the logits before t the same to within 1e-6, for both attention implementations and both position encodings. A model that ignored its context would pass that, so a second test checks that changing the first token does change the last logits.
- **The model uses positions.** With one layer and no position information, attention sees the earlier tokens as an unordered set, so swapping the first two tokens could not change the last position's logits. With either position encoding the swap does change them, and with the positions switched off (a rotation by zero, or a zeroed table) it does not, to within fp64 rounding. I checked that the test fails if rotary is never applied, if it is applied to the queries only, or if the learned table is never added.
- **Attention implementations agree.** `sdpa` and the matmul version give the same logits from the same weights, in fp32.
- **Rotary embeddings depend only on distance.** The score between a query at position m and a key at n is the same as between m + 40 and n + 40.
- **Overfitting.** The model memorizes a 33-token sequence in 150 steps and replays it exactly by greedy decoding from the first token. That shows it can learn; a model without positions could pass it too, which is what the test above is for.
- **The tokenizer.** Random text including accents, curly quotes, CJK and emoji round-trips, and the fast encoder produces exactly the tokens that applying the merges in training order would.
- **Training.** The learning rate schedule; batches being shifted windows of the file, with every window reachable; validation loss falling on a learnable toy dataset; exact resume after a kill partway through the schedule; an update that turns the weights into NaN not overwriting the last checkpoint; gradient clipping; weight decay on the matrices and not the layer norms; the bits-per-byte conversion; logged speed and training time, with a fake clock, leaving out evaluation time; and the full validation pass covering every window once.

## How to run

```
python3 -m venv --system-site-packages .venv    # reuses an installed torch
.venv/bin/pip install torch numpy tiktoken matplotlib pytest
scripts/get_data.sh                             # TinyStoriesV2-GPT4 train and valid, about 2.2 GB
.venv/bin/python prepare.py                     # tokenizer, then data/train.bin and data/val.bin
scripts/full_runs.sh main                       # lid open, power connected
.venv/bin/python sample.py runs/main/ckpt.pt --prompt "Once upon a time, a little fox" --num 3
```

Other entry points:

```
.venv/bin/python train.py configs/main.json --out_dir runs/try --max_steps 200    # any config field can be a flag
.venv/bin/python evaluate.py runs/main/ckpt.pt
scripts/bench.sh precision attention     # timing groups; bench.sh lists them
.venv/bin/python scripts/bench_table.py  # results/bench.jsonl as a table
.venv/bin/python scripts/profile_step.py
.venv/bin/python scripts/attention_precision.py
.venv/bin/python scripts/compare_tokenizers.py
```

`data/`, `runs/` and `.venv/` are not in git. `scripts/get_data.sh` downloads only the two GPT-4 text files from the TinyStories repository on Hugging Face, not the rest of it.

## Limits

- Small and narrow on purpose. A model trained on TinyStories writes TinyStories; it knows nothing else.
- One device, no data parallelism. Gradient accumulation is there as a flag, but none of the configs use it.
- No dropout. The main run samples 106M tokens at random from 552M, so it rarely sees the same text twice and has little to overfit.
- Sampling reruns the whole context for each new token instead of caching keys and values. For a few stories of a few hundred tokens that is fast enough, so I left it simple.
- The rotary variant only ever sees positions up to 256 in training, and sampling past that slides the window rather than extending it.
- Step times were measured on one laptop on battery, with other programs open. The ranges in the table show how much that moved them, and several comparisons rest on one or two runs.
- No separate test split (see above); the final numbers are on the same validation split the runs were watched on.
