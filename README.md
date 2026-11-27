# DeepSeek-V4.1 on a 2-socket CPU server

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository runs [DeepSeek-V4.1-Flash](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (`UD-Q4_K_XL`, 748 billion parameters) **on CPUs only**, on one server: 2 x 16 Xeon cores, 1 TB of RAM, RHEL 8.10. It is a fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) with what that took: changes to the engine and to the server, ready-made packages for RHEL 8, and the scripts that put the model in memory, start the server and measure it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="media/deepseek-before-after-dark.svg">
  <img src="media/deepseek-before-after-light.svg" width="920" alt="Before and after on the 2-socket server. Generating: 9.4 tokens per second before, 22.9 after. Generating with the DSpark draft: 25.6 before, 29.6 after. Reading a prompt: 236 tokens per second before, 361 after. Starting the server again with a conversation of 10,547 tokens: 29 seconds to read it again before, 0.5 second after. A request larger than the context: read for 6 to 7 minutes before, refused at once after. Agents and tools: a model could not write a Metabase query before; tool calls, DeepSeek's API and DeepSeek Harness work after.">
</picture>

The same as a table:

| | Before | After |
|---|---|---|
| Generating, without a draft | 9.4 tokens/s | 22.9 tokens/s |
| Generating with the DSpark draft | 25.6 tokens/s | 29.6 tokens/s (27.2 to 33.6 with the subject) |
| Reading a prompt | 236 tokens/s | 361 tokens/s, and 452 with `GGML_AMX_ATTN=1` (see below) |
| Starting the server again, a conversation of 10,547 tokens | read again from its first token, 29 s (computed from the reading speed) | back from a RAM disk in 0.5 s |
| A request larger than the context | refused, then read all the same for 6 to 7 minutes | refused at once |
| Agents and tools | a model could not write a Metabase query | tool calls, DeepSeek's API and DeepSeek Harness work |

Everything is measured on that one server. The build and the conditions of each figure are in [docs/deepseek/releases.md](docs/deepseek/releases.md). The diagram is drawn by `scripts/before-after.py`.

## Run it

What it takes: a machine with two sockets and about 780 GiB of free memory for this model (two copies of the weights of 277 GiB, one per socket, the engram table of 189 GiB, the drafter), Linux x64 with glibc 2.28 or newer, and a processor with AVX2. AVX-512 and Intel AMX are used where the processor has them.

1. Take the package of the [latest release](https://github.com/muller-camille/ik_llama.cpp/releases/latest). Nothing has to be installed:

   ```
   curl -L -O https://github.com/muller-camille/ik_llama.cpp/releases/download/rhel8-b5404/ik_llama-rhel8-b5404-bin-rhel8-x64.tar.gz
   tar xzf ik_llama-rhel8-b5404-bin-rhel8-x64.tar.gz
   cd ik_llama-rhel8-b5404
   ```

2. Repack the model files once, and the drafter. The repacked files are the ones the server maps from the RAM disks:

   ```
   ./llama-quantize --repack --keep-split DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf \
       /models/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8.gguf COPY
   ./llama-quantize --repack dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
       /models/dspark-DeepSeek-V4.1-Flash-1017-Q8_0-R8.gguf Q8_0_R8
   ```

3. Say what is specific to the machine in `~/.deepseek.conf`, one `NAME=value` per line:

   ```
   NFS=/models
   API_KEY=choose-a-key
   HOST=0.0.0.0
   CTX=262144
   CACHE_DIR=/mnt/dscache/cache
   ```

4. Start it, with your usual account (it asks for `sudo` for the mounts only):

   ```
   ./deepseek.sh
   ```

   The script mounts a RAM disk per socket and copies the weights there if they are not there yet, starts the server with one thread per core and the DSpark draft, and reads prompts in batches of 2048 tokens. With `CACHE_DIR` it also mounts a RAM disk for the conversations the server sets aside, where it finds them again after a restart. Stop it with one Ctrl+C: it finishes the answer in progress and, with `CACHE_DIR`, saves the conversation it holds. The messages of the scripts are in French.

| Setting | Default | What it is |
|---|---|---|
| `NFS` | none, required | folder of the repacked model files and of the drafter |
| `API_KEY` | none | key the clients send; required when `HOST` is not this machine only |
| `HOST`, `PORT` | `127.0.0.1`, `8080` | where the server listens; `0.0.0.0` is every network interface |
| `P`, `M` | `DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8` | model to load, and model to run if it is another one |
| `DRAFT`, `SPEC` | the repacked drafter if it is in `NFS`, `dspark:n_max=4` | drafter file (empty: none) and its setting |
| `T`, `TB` | one thread per physical core | threads, and threads for batches of tokens if they differ |
| `CTX` | `131072` | context size in tokens |
| `UBATCH` | `2048` | tokens of a prompt read together |
| `GGML_AMX_ATTN` | off | `1` sends the attention of a prompt through Intel AMX too: a prompt of 12,616 tokens read at 452 tokens/s instead of 359. The answer to a request is then another text, see [Not measured yet](#not-measured-yet) |
| `CACHE_DIR`, `CACHE_DISK` | empty, `120G` | folder of the saved conversations on a RAM disk (`/mnt/dscache/cache`), and the size of that RAM disk. The conversations are found again after a restart of the server, with another `CTX` too if they fit in it and `UBATCH` is the same; not after a reboot of the machine |
| `WEBUI` | `llamacpp` | web page of the server: `llamacpp`, `auto` or `none` |

What a package holds, every script, and the same setup by hand: [docs/deepseek/setup.md](docs/deepseek/setup.md). DeepSeek's own agent on this server: [docs/deepseek/harness.md](docs/deepseek/harness.md).

## Measure it

Nothing is assumed about another machine: the defaults of `deepseek.sh` come from the script below, run on this server, and it is the one to run again elsewhere.

```
./profil.sh                                               # every pass: about 2 hours, with the usual server stopped
env RAPIDE=1 ./profil.sh                                  # the short version, about 25 minutes
env DRAFT= THREADS=32 VARS=GGML_AMX_ATTN=1 ./profil.sh    # one setting against the reference
env SANS_DRAFTER=0 CONTEXTE=0 VARS="SPEC=dspark:n_max=4,p_min=0.5 SPEC=dspark:n_max=4,p_min=0.3" ./profil.sh    # settings of the draft
```

It measures the thread counts, without and with the draft, after a short and after a long prompt, then one setting changed at a time, and writes `profil-resultat.txt` with a table of the passes and the time of every kind of node of the model. For a server started by hand, `GGML_PROFILE=1` in its environment prints the same node times when it stops. More in [docs/deepseek/measurements.md](docs/deepseek/measurements.md).

## What the fork changes

One line per change. The figures are tokens per second with `rhel8-b5379` after a prompt of 12,616 tokens: 21.5 generated and 361 read with everything on. The runs of one build differ by up to 3%.

| Where | Change | With it off | Switch |
|---|---|---|---|
| Memory | One copy of the weights per socket; each thread reads the copy of its own socket | 9.4 instead of 17.1, when it came | `--numa-mirror DIR` |
| Threads | One thread per physical core | 20.1 with one per logical CPU | `T` |
| Weights | They are asked for ahead of their use, 4096 bytes ahead: seven other distances and cache levels were tried, none is faster | 18.9 | `IQK_PREFETCH=0` |
| Weights | The hyper-connections of DeepSeek-V4 in one node (`hc_proj`) | 19.9 | `GGML_HC_PROJ=0` |
| Weights | The small nodes of a graph on one thread; rows shared between threads (since `rhel8-b5397` in the shared expert too: 2.91 ms of a token instead of 3.09) | 21.3, 21.4 | `GGML_SERIAL_NODES=0`, `GGML_SHARE_ROWS=0` |
| Attention | Only the rows a query sees, on a grid of threads; each token of a draft on its own rows | 19.6 | `GGML_FA_SKIP_MASKED=0` |
| Attention | The cache of DeepSeek-V4 read in its two parts, not copied together | 21.1 | `GGML_LAZY_CONCAT=0` |
| Prompt | Batches of 2048 tokens instead of 512 | 267 read | `UBATCH=512` |
| Prompt | The matrix products of a batch through Intel AMX | 326 read | `GGML_AMX=0` |
| Prompt | The indexer of DeepSeek-V4 in AVX-512 | 347 read | `GGML_INDEXER_512=0` |
| Prompt | The attention of a batch through Intel AMX, on request | 359 read, 452 with it (`rhel8-b5390`) | `GGML_AMX_ATTN=1` turns it on |
| Server | The conversations it sets aside are files on a RAM disk: still there after a restart or a crash, and after a change of the context size | read again from the first token | `--cache-ram-dir DIR`, `CACHE_DIR` |
| Server | A request larger than the context is refused without being read | read for 6 to 7 minutes | |
| Clients | The fields of DeepSeek's API, tool arguments as JSON Schema means them, Unsloth's chat template recognized | a model could not write a Metabase query | |
| Loading | The GGUF files of upstream llama.cpp load as they are, engram table and DSpark draft included | | |
| Draft | The DSpark draft with its own settings (4 tokens at most, stopped at the first one it is less than 75% sure of): five other settings were measured, none is faster | after a short prompt, 30.4 with it (`rhel8-b5399`) and 23.1 without (`rhel8-b5397`) | `SPEC`, `DRAFT=` |
| Tools | The time of every kind of node, and how fast the weights are read | | `GGML_PROFILE=1` |
| Packages | CPU-only builds for RHEL 8 (`avx2`, `avx512`, `avx512-bf16`) with the scripts | | |

What each change does, how it was checked and what else was measured: [docs/deepseek/changes.md](docs/deepseek/changes.md). What each release changed, with the charts of the speed release by release: [docs/deepseek/releases.md](docs/deepseek/releases.md).

## Not measured yet

- What the attention through Intel AMX (`GGML_AMX_ATTN=1`) does to the answers. Its speed is measured: a prompt is read 26% faster. It rounds queries, cache rows and attention weights to 8 significant bits, 0.26% from the exact result on random numbers, and the text generated after the same prompt is another one. Whether the answers are as good has not been compared, which is why it is not the default.
- The non-expert tensors in `Q6_K` instead of `Q8_0` (`recompresser.sh`): fewer bytes to read per token, estimated at 8 to 12% for the generation, and it changes the model.

## More

- [docs/deepseek/changes.md](docs/deepseek/changes.md): every change in detail
- [docs/deepseek/releases.md](docs/deepseek/releases.md): release by release, with the figures and the charts
- [docs/deepseek/measurements.md](docs/deepseek/measurements.md): settings compared on one build, where the time goes
- [docs/deepseek/setup.md](docs/deepseek/setup.md): the packages, the scripts, the setup by hand
- [docs/deepseek/harness.md](docs/deepseek/harness.md): DeepSeek Harness on this server
- [README-ik_llama.cpp.md](README-ik_llama.cpp.md): the README of ik_llama.cpp itself (building, quantization types, features, options)
