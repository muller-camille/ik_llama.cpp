# Measurements on the 2-socket server

[README](../../README.md) · [changes](changes.md) · [releases](releases.md) · [setup](setup.md) · [DeepSeek Harness](harness.md)

The figures of each release are in [releases.md](releases.md). Here: the measurements that compare settings on one build, how to see where the time goes, and the one lever left once the memory sets the speed.

## One build, several settings

Measured on a server with 2 x 16 Sapphire Rapids cores and 8 DDR5 channels per socket, 200 tokens generated after a short prompt:

| Build | Threads | Draft | Tokens/s |
|---|---|---|---|
| `rhel8-b5301` | 64 | | 18.4 |
| `rhel8-b5308` (serial nodes, prefetch) | 64 | | 20.0 |
| `rhel8-b5308` | 32 | | 21.3 |
| `rhel8-b5317` (`hc_proj`, even row split, row sharing) | 64 | | 20.8 |
| `rhel8-b5317` | 32 | | 22.6 |
| `rhel8-b5301` | 64 | DSpark, `n_max=4` | 26.0 |
| `rhel8-b5308` | 64 | DSpark, `n_max=4` | 25.9 |
| `rhel8-b5317` | 32 | DSpark, `n_max=4` | 30.3 |

`rhel8-b5340` (attention sinks for a token decoded alone, attention on the rows in view), 32 threads:

| Prompt | Draft | Tokens/s | Attention per graph |
|---|---|---|---|
| short | | 22.9 | 1.9 ms |
| short | DSpark, `n_max=4` | 31.1 | 2.9 ms for 3.3 tokens |
| 3159 tokens | | 21.4 | 2.7 or 5.6 ms |

With a context of a few thousand tokens the attention is the largest cost that is not reading weights. The text differs from `rhel8-b5317` because of the sinks, and with it the share of accepted draft tokens (63% here, 74% there): a draft step takes 74.4 ms instead of 78.3 ms, for about the same tokens/s on these texts.

`rhel8-b5342` (the threads of the attention in a grid, groups of rows by groups of heads), 32 threads, and the same without groups of rows:

| Prompt | Draft | Tokens/s | Attention per graph | `GGML_FA_ROW_GROUPS=1` |
|---|---|---|---|---|
| short | | 23.1 | 1.5 ms | 22.6, 1.9 ms |
| 3165 tokens | | 21.7 | 2.9 ms | 20.6, 5.1 ms |
| short | DSpark, `n_max=4` | 30.9 | 3.0 ms for 3.4 tokens | 30.9, 3.1 ms |
| 3165 tokens | DSpark, `n_max=4` | 24.5 | 8.9 ms for 3.0 tokens | |

In the last line a draft step takes 67.2 ms, and 58% of the draft tokens were accepted on this text.

`rhel8-b5345` (the lists of a draft on the grid of threads), 32 threads:

| Prompt | Draft | Tokens/s | Attention per graph |
|---|---|---|---|
| short | | 23.1 | 1.5 ms |
| 3165 tokens | | 21.9 | 2.9 to 3.0 ms |
| short | DSpark, `n_max=4` | 31.5 | 3.0 ms for 3.4 tokens |
| 3165 tokens | DSpark, `n_max=4` | 27.7 | 7.2 ms for 3.0 tokens |

In the last line a draft step takes 64.4 ms instead of 67.2, and 68% of the draft tokens were accepted on this text (58% in the run of `rhel8-b5342`, hence most of the difference in tokens/s). The same release with the number of row groups drawn for every node (`GGML_FA_ROW_GROUPS=-1`), time of thread 0 in an attention node, in microseconds:

| Row groups | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| one token, 672 rows read in place | 99 | 73.6 | 59.5 | 65.1 |
| about 3 tokens of a draft, 640 rows of each gathered | 258 | 210.6 | 171.7 | 186.5 |

With one row group a thread does not wait for the others, so its time understates the node. Asking for the rows in advance before copying them made no difference (171.4 against 172.0 with 4 row groups) and was removed. The choice of the number of row groups goes by these figures since `rhel8-b5349`.

With `rhel8-b5317` and 32 threads, one optimization turned off at a time:

| Switch | Tokens/s | With the draft |
|---|---|---|
| none | 22.6 | 30.3 |
| `GGML_SHARE_ROWS=0` | 22.2 | 30.1 |
| `GGML_HC_PROJ=0` | 20.5 | 27.3 |

Each figure is the mean of 5 runs of 200 tokens, and the runs of one build differ by up to 3%, so the 2% of the row sharing is at the limit of what this shows. The draft was accepted at 74% in these runs (68% in the run with `GGML_HC_PROJ=0`, whose text differs), so its gain depends on the text.

## Where the time goes

```
GGML_PROFILE=1 ./llama-server -m model.gguf ... 2> server.log    # stop it with Ctrl+C
grep ggml_profile server.log
```

A line per kind of node, the layers added up: milliseconds per graph, share, how busy the threads were (all, the even ones, the odd ones, the least and the most busy one) and the GB/s of weights read. For the expert nodes these are the weights of the experts the tokens of a graph picked, an expert that several tokens picked counted once, since it is read once (before `rhel8-b5390` it was counted once per token, which made the experts of a graph of several tokens look read faster than they were: 543 GB/s for the 3.3 tokens of a draft verification in the run of `rhel8-b5371`, against 331 for a token alone). A node whose threads are far from 100% busy is waiting for something else than its own work.

## Fewer bytes per token

Once the memory sets the speed, the remaining lever is to read less. The `Q8_0` tensors of `UD-Q4_K_XL` that are not experts (attention, shared expert, output) are 7.9 of the 12.9 GB read per token. This requantizes them to `Q6_K` and writes only the last split file again, 7.3 GB instead of 9.2:

```
# in the output directory, link the files 1-8 under the new name first: they are not written again
./llama-quantize --allow-requantize --keep-split --partial-requant \
    --custom-q '^(blk\.[0-9]+\.(attn_kv|attn_q_[ab]|attn_output_[ab]|ffn_(up|gate|down)_shexp|engram_kv|attn_compressor_(gate|kv)|indexer\.(proj|attn_k|attn_q_b))|output)\.weight$=q6_k_r4' \
    DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8-00001-of-00009.gguf DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8Q6.gguf COPY
```

This changes the model. On random weights the error of these tensors goes from 0.5% (`Q8_0`) to 1.8%; **the effect on the quality of the answers is not measured**, and neither is the gain on the 2-socket server (estimated at 8 to 12%).
