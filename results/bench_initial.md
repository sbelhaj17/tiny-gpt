| setting | params | ms per step | spread | tokens/s | GPU memory |
|---|---|---|---|---|---|
| dtype=fp32, attn=sdpa | 12.2M | 432 | 1 run | 18,962 | 3.52 GB |
| attn=sdpa | 12.2M | 447 | 445-453 (8 runs) | 18,321 | 3.53 GB |
| dtype=fp16, attn=sdpa, optimizer=foreach | 12.2M | 459 | 1 run | 17,846 | 3.53 GB |
| attn=sdpa, optimizer=foreach | 12.2M | 451 | 451-452 (2 runs) | 18,150 | 3.53 GB |
| attn=sdpa, optimizer=loop | 12.2M | 448 | 1 run | 18,272 | 3.53 GB |
| baseline | 12.2M | 410 | 1 run | 19,967 | 3.48 GB |
| attn=sdpa, compile=True | 12.2M | 509 | 1 run | 16,080 | 2.3 GB |
| attn=sdpa, same_batch=True | 12.2M | 447 | 1 run | 18,330 | 3.6 GB |
| batch_size=8, attn=sdpa | 12.2M | 116 | 1 run | 17,581 | 1.28 GB |
| batch_size=16, attn=sdpa | 12.2M | 224 | 1 run | 18,262 | 2.43 GB |
| batch_size=64, attn=sdpa | 12.2M | 913 | 1 run | 17,953 | 5.86 GB |
| batch_size=128, attn=sdpa | 12.2M | 1827 | 1 run | 17,933 | 8.92 GB |
| attn=sdpa, vocab_size=50257 | 29.93M | 1085 | 1 run | 7,551 | 11.39 GB |
| n_layer=4, n_embd=256, attn=sdpa | 4.2M | 191 | 1 run | 42,821 | 2.43 GB |
| n_layer=8, n_embd=512, attn=sdpa | 27.28M | 876 | 1 run | 9,353 | 4.71 GB |
