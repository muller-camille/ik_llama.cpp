# Setup: the packages, the scripts, and the same by hand

[README](../../README.md) · [changes](changes.md) · [releases](releases.md) · [measurements](measurements.md) · [DeepSeek Harness](harness.md)

## The packages

The [releases](https://github.com/muller-camille/ik_llama.cpp/releases) named `rhel8-b<number>` are CPU-only builds for RHEL 8.10 (Linux x64, glibc 2.28). A package holds three static builds (`avx2`, `avx512`, `avx512-bf16`) and launchers that start the one matching the CPU; nothing has to be installed. The workflow is `.github/workflows/release-rhel8.yml`.

```
curl -L -O https://github.com/muller-camille/ik_llama.cpp/releases/download/rhel8-b5379/ik_llama-rhel8-b5379-bin-rhel8-x64.tar.gz
tar xzf ik_llama-rhel8-b5379-bin-rhel8-x64.tar.gz
./ik_llama-rhel8-b5379/llama-server --version
```

A package also holds the shell scripts used on the 2-socket machine (`scripts/deepseek` in this repository; their messages are in French). They use the tools next to them, and read what is specific to the machine (`NFS`, `API_KEY`, `HOST`, ...) from `~/.deepseek.conf`, one `NAME=value` per line:

| Script | What it does |
|---|---|
| `deepseek.sh` | loads the model into RAM, one copy of the weights per socket, and starts the server; with the draft model, its repacked file (`llama-quantize --repack <drafter> <drafter>-R8.gguf Q8_0_R8`) when that is in the folder of the model files: a draft call took 9.5 ms instead of 11.5 with it. The prompt is read in batches of 2048 tokens (`UBATCH`, the server's `-ub`; the server's own default is 512). With `--swa-compress` the 19 decoder layers of DeepSeek-V4.1 are computed for the last 256 tokens of a batch only, so the larger the batch, the smaller the share of the prompt that goes through them: one token in eight instead of one in two. The expert weights, which are read from the memory once per batch, serve four times the tokens as well. On the 2-socket server a prompt of 15761 tokens is read at 329 tokens/s in batches of 2048 against 245 in batches of 512, and the text after it is generated at the same speed. `UBATCH=512` goes back to the server's default. Threads (`T`): one per physical core, with or without the draft model, the faster of the two in the run of `rhel8-b5379` on the 2-socket server, where every figure is a mean over five subjects: 22.9 tokens/s against 21.5 with one per logical CPU without the draft, 29.6 against 28.3 with it, 24.2 against 23.6 after a prompt of 12616 tokens, and the prompt read at the same speed (361 and 358 tokens/s). With the draft the result moves with the text: on the subject of the earlier runs one per logical CPU stays ahead (31.5 against 29.8, and 33.4 against 30.9 with `rhel8-b5371`), on three of the four others one per core is. `TB` sets the threads for batches of tokens, the server's `-tb`; with the draft it is to be left equal to `T`: at 32/64 a step of the draft takes 94 ms instead of 83. `MMPROJ=mmproj-F16.gguf` loads the image encoder of the model, a file of 926 MiB from its repository that goes into the folder of the model files as it is: the script copies it to the RAM disk of socket 0 and gives it to the server (`--mmproj`), which then accepts images; without it an image is refused. Not tried on the 2-socket server yet |
| `charger-deepseek.sh` | the loading alone, as root |
| `mesure.sh` | starts a server of its own and measures the speed, with or without a draft |
| `profil.sh` | the measurement the defaults of `deepseek.sh` come from, with `GGML_PROFILE=1`, in one run: without and with the drafter it finds next to the model, after a short prompt and after one of 16000 tokens, with one thread per physical core, then one per logical CPU (`THREADS="32 64 32/64"` adds 32 threads for a single token and 64 for batches of tokens, the server's `-t` and `-tb`); then one setting changed at a time (`VARS`: without AMX, with AMX for the attention too, batches of 512 and of 4096 tokens, and every optimization of the table above that has a switch), with the threads `deepseek.sh` starts. A setting of the draft is compared the same way (`SPEC=dspark:n_max=4,p_min=0.5` among `VARS`, with the draft only; `SANS_DRAFTER=0` leaves out the passes without it). A pass makes 5 requests, each for a text on another subject (the first is the subject of every earlier run), and after the long prompt 3 of them read a new text, which the server cannot take from its cache. 32 passes and about 2 hours on 2 x 16 cores with hyperthreading, with the usual server stopped; `RAPIDE=1` makes 14 in 25 minutes, and Ctrl+C keeps what is done. The result file ends with a table, one line per pass (tokens read, reading speed, generation speed with its lowest and highest try, share of the draft accepted, and for the passes with the drafter the duration of one of its steps and the tokens a step gives), then the fastest thread count column by column with the figures of the others, and what each setting of `VARS` changes next to the pass without it |
| `drafter.sh` | tries several settings of the DSpark draft |
| `recompresser.sh` | requantizes the non-expert `Q8_0` tensors and installs the result next to the model |
| `harness.sh` | starts [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`: an agent with a web UI and a command line) with this server in the place of DeepSeek's API, see [harness.md](harness.md) |

## Using a package directly

What the notes of every release used to say; the notes now hold only what a release changes.

A package is a CPU-only build for RHEL 8.10 (Linux x64, glibc 2.28), without a GPU backend:

- Three static builds: `avx2` (Haswell, Zen 1 and newer), `avx512` (Cascade Lake, Ice Lake, Zen 4 and newer), `avx512-bf16` (Sapphire Rapids, Zen 4 and newer).
- `llama-server`, `llama-cli`, ... in the top directory are launchers: they start the build that matches the CPU. Set `IK_LLAMA_VARIANT` to pick one by hand.
- No extra packages are needed, the OpenMP runtime is in `lib/`.
- A build made on the machine itself with `-DGGML_NATIVE=ON` can still be faster.

DeepSeek-V4.1 from an upstream llama.cpp GGUF (for example `unsloth/DeepSeek-V4.1-Flash-1017-GGUF`) loads as it is, MXFP8 engram table included:

```
./llama-server -m UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf --jinja --swa-compress
```

Speculative decoding with the DSpark draft of the same repo is optional and does not change the output; the server log prints the draft acceptance rate:

```
./llama-server -m UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf \
    -md dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf --spec-type dspark:n_max=3 --jinja --swa-compress
```

To use a smaller engram table, stamp it with the quant's table slot and pass it with `--ngram-file`; the quant's own table shard is then not needed:

```
python engram/set_split.py engram/DeepSeek-V4.1-Flash-1017-engram-Q6_K.gguf_file 8 9
./llama-server -m UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf \
    --ngram-file engram/DeepSeek-V4.1-Flash-1017-engram-Q6_K.gguf_file --jinja --swa-compress
```

On a machine with several sockets and enough RAM, keep one copy of the weight shards per NUMA node, each on a tmpfs bound to its node, and pass the extra copies with `--numa-mirror`. Every thread then reads the weights from its own node. The table shard can stay shared:

```
mount -t tmpfs -o size=285G,huge=always,mpol=bind:0 tmpfs /mnt/ds0
mount -t tmpfs -o size=285G,huge=always,mpol=bind:1 tmpfs /mnt/ds1
# shards 1-7 and 9 in both directories, shard 8 (the table) linked into /mnt/ds0
./llama-server -m /mnt/ds0/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf --numa-mirror /mnt/ds1 --jinja --swa-compress
```

The log tells which node holds each copy. `--numa-mirror` needs the weights mapped from the files, so it does not work with `-rtr`: repack the files once with `llama-quantize --repack --keep-split` instead.

To see where the time goes, start a tool with `GGML_PROFILE=1` in the environment. When it exits, it prints for every kind of node the elapsed time per token, how busy the threads were and how fast the weights were read:

```
GGML_PROFILE=1 ./llama-server -m model.gguf ... 2> server.log    # stop it with Ctrl+C, then: grep ggml_profile server.log
```

On a Xeon with AMX (Sapphire Rapids and later), a batch of 48 tokens or more is multiplied with the `Q8_0_R8` matrices through the AMX tiles, as bf16: about twice as fast for those products, which are a fifth of the reading of a prompt with DeepSeek-V4.1. `grep amx /proc/cpuinfo` tells whether the system gives AMX to programs; `GGML_AMX=0` in the environment turns it off. `GGML_AMX_ATTN=1` sends the attention of a prompt batch through the tiles too (DeepSeek-V4: a third of the reading of a prompt). Measured on the 2-socket server: a prompt read at 452 tokens/s instead of 359. It is not the default: queries, cache rows and attention weights are rounded to bf16 there, the text generated after the same prompt is another one, and whether the answers are as good has not been compared. With `deepseek.sh`, a line `GGML_AMX_ATTN=1` in `~/.deepseek.conf` turns it on: the script gives every `NAME=value` line of that file to the server as an environment variable.

`llama-server --cache-ram-dir DIR` keeps the prompt cache (the conversations the server sets aside when another one comes) as files in DIR instead of in its own memory. With DIR on a RAM disk they are still there after the server is restarted, and the conversation in each slot is added when the server stops. Off unless asked for.

`llama-server --webui llamacpp` serves the current web UI of llama.cpp ("Llama Chat"); without the option it serves the one of ik_llama.cpp.

## By hand, without the scripts

For one token DeepSeek-V4.1-Flash (`UD-Q4_K_XL`) reads 12.9 GB of weights, and they do not stay in any cache. The speed is the speed of the memory, so the set-up is about where the weights are and which thread reads them.

1. Repack the files once. `--numa-mirror` maps the weights from the files, so `-rtr` cannot be used with it:

   ```
   ./llama-quantize --repack --keep-split DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf \
       /models/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8.gguf COPY
   ```

2. Keep one copy of the weight files per NUMA node in RAM, each on a tmpfs bound to its node. The engram table (file 8 of 9, 200 GB) is read sparsely and can stay shared: link it into the first directory only.

   ```
   mount -t tmpfs -o size=300G,huge=always,mpol=bind:0 tmpfs /mnt/ds0
   mount -t tmpfs -o size=300G,huge=always,mpol=bind:1 tmpfs /mnt/ds1
   # files 1-7 and 9 in both directories, file 8 linked into /mnt/ds0
   ```

3. Start with one thread per physical core. On this machine 32 threads are faster than 64: the weights arrive at the same rate either way.

   ```
   numactl --interleave=all ./llama-server \
       -m /mnt/ds0/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-R8-00001-of-00009.gguf --numa-mirror /mnt/ds1 \
       -t 32 -c 131072 --jinja --swa-compress
   ```

   The log tells which node holds each copy.

4. Optionally add the DSpark draft of the same repository. It does not change the output; the log prints the acceptance rate:

   ```
       -md /mnt/ds0/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf --spec-type dspark:n_max=4
   ```
