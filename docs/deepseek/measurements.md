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

## The attention through AMX, the prefetch distances, the experts (`rhel8-b5390`)

32 threads, no draft, after a prompt of 12,616 tokens; 3 tries per pass, 2 of which read a new text. One setting changed at a time:

| Setting | Prompt read, tokens/s | Generated, tokens/s (lowest and highest try) |
|---|---|---|
| none | 358.6 | 21.72 (21.19, 22.05) |
| `GGML_AMX_ATTN=1` | 452.2 | 21.73 (21.48, 21.99) |
| `IQK_PREFETCH=1024` | 352.6 | 20.37 (20.05, 20.65) |
| `IQK_PREFETCH=16384` | 356.2 | 21.36 (21.04, 21.56) |
| `IQK_PREFETCH=4096,32768` | 357.5 | 21.46 (21.18, 21.70) |
| `IQK_PREFETCH=4096,131072` | 357.7 | 20.90 (20.59, 21.28) |
| `IQK_PREFETCH=1024,16384` | 354.6 | 21.42 (21.15, 21.68) |
| `IQK_PREFETCH=16384,0,1` | 357.1 | 21.66 (21.55, 21.78) |
| `IQK_PREFETCH=65536,0,1` | 351.7 | 20.66 (20.46, 20.79) |

After a short prompt the same build generates 22.78 tokens/s (22.52, 22.99).

**The attention through AMX** reads the prompt 26% faster, and the two readings agree (452.18 and 452.25). In the profile a batch graph, 1679 tokens on average, takes 3659 ms instead of 4648, and its attention 567 ms instead of 1559, 2.75 times less. The generation does not move: a token generated alone does not go this way. The text generated after the same prompt is another one, where the seven passes with another prefetch setting all give the text of the reference, character for character. That is what a change of rounding does, and it says nothing about which text is better: that comparison has not been made.

What a batch is made of once the attention goes through AMX:

| Part of a batch graph | ms | Share |
|---|---|---|
| The experts: `ffn_up_exps` with its gate, then `ffn_down_exps` | 1024 + 566 | 43.5% |
| The `Q8_0` matrices that go through AMX (`attn_q_a`, `attn_q_b`, `attn_kv`, `attn_output_a`, `attn_output_b`, `ffn_down_shexp`, `engram_kv`) | 634 | 17.3% |
| The attention | 567 | 15.5% |
| Up and gate of the shared expert (`ffn_up_shexp`), not through AMX | 158 | 4.3% |
| The `f32` matrices (`ffn_gate_inp`, `hc_attn_fn`, `hc_ffn_fn`) | 200 | 5.5% |
| Everything else (norms, hyper-connections, indexer, RoPE...) | 510 | 13.9% |

**The prefetch**: the kernels ask for the weights 4096 bytes ahead, into the L1 cache. No other setting is faster. Asking nearer (1024), much farther (65536), or a second time from 131072 bytes ahead loses 4 to 6%; the four others are within the 3% that the runs of one build differ by. Requests sent to the L2 cache (`16384,0,1`) give the same speed as the default, so nothing here shows that a core can be made to read more than it does.

**The experts**, with the count of this build (an expert that several tokens picked is counted once): for a token alone the up and gate matrices are read at 326 GB/s and the down matrices at 320; for the 5 tokens that end a prompt, at 351 and 337 GB/s. The 543 GB/s that the profile of `rhel8-b5379` showed when a draft is verified came from counting an expert once per token. For comparison `output.weight`, one matrix of 0.7 GB read from end to end, is read at 450 GB/s.

## Rows shared in the experts (`rhel8-b5397`)

32 threads, no draft, 3 tries per pass. `rhel8-b5397` let the threads take rows from each other in the shared expert and at the end of the chunks of the routed experts. Time of these nodes for a token (the 40 layers), and how busy the threads were; after the long prompt both builds have the attention through AMX and generate the same text:

| Node, one token per graph | `rhel8-b5390` | `rhel8-b5397` | `rhel8-b5397`, `GGML_SHARE_ROWS=0` |
|---|---|---|---|
| Shared expert, up and gate (`ffn_up_shexp`) | 3.09 ms, 83% | 2.91 ms, 93% | 3.09 ms, 83% |
| Routed experts, up and gate (`ffn_up_exps`) | 9.26 ms, 91% | 9.03 ms, 96% | 9.14 ms, 92% |
| Routed experts, down (`ffn_down_exps`) | 4.69 ms, 93% | 4.80 ms, 97% | 4.73 ms, 92% |
| The whole graph | 45.46 ms | 45.48 ms | 45.90 ms |
| Tokens/s after the prompt of 12,616 tokens | 21.73 | 21.73 | 21.55 |

After a short prompt: 3.22 to 3.08 ms for the shared expert, 9.19 to 9.09 and 4.86 to 4.92 for the routed experts, 43.52 to 42.90 ms for the graph, 22.78 to 23.11 tokens/s.

- **The shared expert gains**: 0.14 to 0.19 ms per token, below every one of the nine passes of `rhel8-b5390` (3.09 to 3.33 ms).
- **The routed experts do not**: their threads are busier, as intended, and the two nodes together take 13.83 ms instead of 13.95, within what two runs differ by; the down matrices are slower in both comparisons. The memory sets the speed there: when some threads are done, the others read faster. This part is taken out again in the next release.
- **Tokens/s do not move** beyond the 3% that the runs of one build differ by. The estimate made from the profile of `rhel8-b5390`, up to 1 ms per token, was too high.
- **The values are the same**: the texts have the fingerprints of `rhel8-b5390`, with the sharing and without it.
- **The attention through AMX**, on in every pass of this run: the prompt read at 455.0 and 454.2 tokens/s (452.2 with `rhel8-b5390`).

Not explained: in the graphs of 5 tokens that end a prompt (2 to 6 per pass), one thread was held up 25 to 65 ms in an expert node, once in each of the three passes, with the sharing on and with it off; the threads were then busy 46 to 86% of that node instead of 97%. None of the ten passes of the run of `rhel8-b5390` shows it in these graphs, where single nodes of the graphs of one token do show such delays (a node at 90% instead of 96 in one pass). With so few graphs it can be chance; the graphs a draft is verified with are of this kind, and a run with the draft has hundreds of them. The run of `rhel8-b5399` with the draft, below, answers it: it is the machine, not this code.

## The settings of the draft (`rhel8-b5399`)

32 threads, with the DSpark draft, after a short prompt; 5 tries of 200 tokens per pass, each on another subject. The drafter proposes up to `n_max` tokens and stops at the first one it is less than `p_min` sure of, 0.75 when nothing is said:

| `--spec-type` | Tokens/s (lowest, highest try) | Proposed per step | Accepted | Produced per step | A step |
|---|---|---|---|---|---|
| `dspark:n_max=4`, as `deepseek.sh` starts it | 30.42 (27.99, 34.43) | 2.16 | 62% | 2.35 | 78.2 ms |
| `dspark:n_max=4,p_min=0.5` | 30.80 (28.58, 33.67) | 2.92 | 53% | 2.58 | 83.9 ms |
| `dspark:n_max=4,p_min=0.3` | 30.05 (24.89, 32.73) | 3.67 | 46% | 2.68 | 89.6 ms |
| `dspark:n_max=4,p_min=0` | 30.60 (28.80, 31.92) | 4.00 | 45% | 2.82 | 92.4 ms |
| `dspark:n_max=5,p_min=0.3` | 30.12 (27.53, 32.93) | 4.38 | 44% | 2.94 | 98.0 ms |
| `dspark:n_max=5,p_min=0` | 28.76 (26.38, 31.22) | 5.00 | 38% | 2.91 | 101.7 ms |

- **No setting is faster than the one in use.** The four nearest are within 1.2%, where a difference of less than 1 token/s between two passes with the draft says nothing; 5 tokens without a threshold is 5.5% slower.
- **Why**: a lower threshold gives more tokens per step and a step that is longer by as much. One more proposed token costs 7.5 to 8.5 ms of verification, a produced token is 33 ms of a step, so it pays when it is accepted more than about one time in four; the tokens the drafter is less than 75% sure of are accepted 15 to 30% of the time.
- The setting of `deepseek.sh` stays.

What a step is made of, in the pass of the setting in use (406 steps):

| Part of a step | ms |
|---|---|
| The verification of the proposed tokens: 37 layers, 3.2 tokens per graph, of which 26.7 ms for the routed experts | 60.9 |
| The last 3 layers, in three graphs of their own, the last one with the logits | 6.9 |
| The drafter: its 3 layers, its logits, and what it takes from the model | 6.8 |
| Outside the graphs: the sampling, the cache, the server | 3.7 |

The output matrix (0.7 GB) is read twice in a step, for the logits of the model and for those of the drafter: 3.3 ms.

**The threads held up in the run of `rhel8-b5397`.** In the 337 to 415 graphs of several tokens of each pass, every thread is busy 95 to 97% of the two nodes of the routed experts: none is held up there. The same pattern, one thread at work and the others waiting, does show in other nodes of some passes: the matrix through which the drafter takes the state of the model (`fc.weight`, busy 63 and 70% in two passes instead of 91 to 95), `attn_q_b` in the small graphs of one pass (60% instead of 95), the shared expert in one pass (84% instead of 92). One thread each time, the same one in two passes. So something on the machine takes a core away for some tens of ms now and then, whatever node the thread is in; it costs about half a percent of a pass at most.

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
