| setting | params | ms per step | spread | tokens/s | GPU memory |
|---|---|---|---|---|---|
| dtype=fp32 | 12.2M | 432 | 418-438 (4 runs) | 18,960 | 3.47 GB |
| baseline | 12.2M | 420 | 408-485 (16 runs) | 19,494 | 3.48 GB |
| dtype=fp16, optimizer=foreach | 12.2M | 429 | 423-435 (2 runs) | 19,107 | 3.48 GB |
| optimizer=foreach | 12.2M | 419 | 416-432 (4 runs) | 19,548 | 3.48 GB |
| optimizer=loop | 12.2M | 425 | 422-428 (2 runs) | 19,276 | 3.48 GB |
| attn=sdpa | 12.2M | 459 | 454-464 (2 runs) | 17,838 | 3.53 GB |
| dtype=fp32, attn=sdpa | 12.2M | 443 | 441-445 (2 runs) | 18,499 | 3.52 GB |
| compile=True | 12.2M | 512 | 508-515 (2 runs) | 16,008 | 2.27 GB |
| same_batch=True | 12.2M | 422 | 422-422 (2 runs) | 19,410 | 3.55 GB |
| batch_size=8 | 12.2M | 109 | 108-110 (2 runs) | 18,786 | 1.27 GB |
| batch_size=16 | 12.2M | 215 | 211-218 (2 runs) | 19,092 | 2.37 GB |
| batch_size=64 | 12.2M | 903 | 843-963 (2 runs) | 18,225 | 4.69 GB |
| batch_size=128 | 12.2M | 1801 | 1686-1916 (2 runs) | 18,272 | 8.18 GB |
| vocab_size=50257 | 29.93M | 1093 | 1044-1141 (2 runs) | 7,512 | 11.34 GB |
| n_layer=4, n_embd=256 | 4.2M | 181 | 177-185 (2 runs) | 45,232 | 2.41 GB |
| n_layer=8, n_embd=512 | 27.28M | 808 | 796-820 (2 runs) | 10,135 | 4.64 GB |
