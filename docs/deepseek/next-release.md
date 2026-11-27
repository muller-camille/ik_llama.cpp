since: rhel8-b5390

**Fewer threads waiting while a token is generated.** Not measured on the server yet: the profile will say.

- **The shared expert** (`ffn_up_shexp`, its up and gate matrices computed together): a thread that is done with its rows takes rows the others have not started, as the other large matrices have done since `rhel8-b5317`. In the profile of `rhel8-b5390` the threads were busy 82% of this node for a token, against 94 to 97% in the matrices that share their rows.
- **The routed experts** (`ffn_up_exps`, `ffn_down_exps`): their chunks were already handed out one by one; a thread that finds none left now takes the end of the chunks the others are still on. The threads were busy 91 to 93% of these nodes.
- These three nodes are 17 of the 45.5 ms of a token. If their threads get as busy as in the other matrices, a token takes up to 1 ms less, 2%: an estimate from the profile, to be measured (`busy` of `FUSED_UP_GATE`, `MOE_FUSED_UP_GATE` and `MUL_MAT_ID` in the result of `profil.sh`).
- Every value is, bit for bit, the one it was with each thread on its own rows: `test-share-rows` compares them for 2 to 64 threads and 1 to 9 tokens, with experts in the format of this model (`MXFP4_R8`) and in `Q8_0_R8`. A prompt batch is computed as before. `GGML_SHARE_ROWS=0` turns the sharing off, to compare.

**Measured with `rhel8-b5390` on the 2-socket server**, now in the README and in [measurements.md](https://github.com/muller-camille/ik_llama.cpp/blob/main/docs/deepseek/measurements.md):

- `GGML_AMX_ATTN=1` reads a prompt of 12,616 tokens at 452 tokens/s instead of 359, and the generation does not change. It stays off unless asked: the text generated after the same prompt is another one, and whether the answers are as good has not been compared. With `deepseek.sh`, a line `GGML_AMX_ATTN=1` in `~/.deepseek.conf` turns it on.
- No other distance or cache level of the weight prefetch (`IQK_PREFETCH`) is faster than the current one.

**Around the binaries**

- **Release notes**: the notes of a release now hold only what it changes, as here. What every release used to repeat (what a package holds, how to start the server by hand) is in [docs/deepseek/setup.md](https://github.com/muller-camille/ik_llama.cpp/blob/main/docs/deepseek/setup.md).
- **README**: rewritten around the project, DeepSeek-V4.1 on a 2-socket CPU server: what it does now against before, how to run it, how to measure it, and one line per change. The long tables moved to `docs/deepseek/` (changes in detail, release by release, measurements, setup, DeepSeek Harness), and the README of ik_llama.cpp is in `README-ik_llama.cpp.md`.
- **Before and after**: a diagram at the top of the README, with the figures measured on the server: 9.4 to 22.9 tokens/s generated, 25.6 to 29.6 with the draft, 236 to 361 tokens/s for a prompt, a conversation of 10,547 tokens back in 0.5 s after a restart.
