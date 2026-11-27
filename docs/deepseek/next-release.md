since: rhel8-b5390

Nothing changes in the binaries: this release is about what is written around them.

- **Release notes**: the notes of a release now hold only what it changes, as here. What every release used to repeat (what a package holds, how to start the server by hand) is in [docs/deepseek/setup.md](https://github.com/muller-camille/ik_llama.cpp/blob/main/docs/deepseek/setup.md).
- **README**: rewritten around the project, DeepSeek-V4.1 on a 2-socket CPU server: what it does now against before, how to run it, how to measure it, and one line per change. The long tables moved to `docs/deepseek/` (changes in detail, release by release, measurements, setup, DeepSeek Harness), and the README of ik_llama.cpp is in `README-ik_llama.cpp.md`.
- **Before and after**: a diagram at the top of the README, with the figures measured on the server: 9.4 to 22.9 tokens/s generated, 25.6 to 29.6 with the draft, 236 to 361 tokens/s for a prompt, a conversation of 10,547 tokens back in 0.5 s after a restart.
