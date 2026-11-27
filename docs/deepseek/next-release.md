since: rhel8-b5399

**The saved conversations no longer belong to one context size.**

- Until now a file of the prompt cache (`--cache-ram-dir`, `CACHE_DIR`) was taken back only by a server with the context size that wrote it: after a change of `CTX` the conversations stayed in the directory, unused.
- Now, for DeepSeek-V4.1 as `deepseek.sh` runs it (`--swa-compress`), the server takes back every conversation that fits in its context, whatever the context it was written with. The files written by `rhel8-b5390` to `rhel8-b5399` are found too.
- `UBATCH` has to be the same as when the files were written: it sets the number of rows of the sliding window in the saved state.
- A conversation that is longer than the new context, or was written with another batch size, is left in the directory for a server that can read it. The start of the server says how many there are and how large: they are not counted in the size of the cache and the server never removes them.
- If a file does not come back in a server that was set up in another way, the prompt is read again, and the file is not tried a second time by that server; it stays until that server has written the conversation anew or needs the room.
- Other models keep the context size in what names their files, as before.
- How it was checked: the saved state of this model was read through in the engine, structure by structure, for what depends on the context size; `test-prompt-store` checks the files (those of earlier versions found, put in order of age, read and removed like the others). The way through the server with the model is not tested: the first restart with another `CTX` is that test, and the log says whether each conversation came back.

Nothing survives a reboot of the machine, as before: `/mnt/dscache` is a RAM disk.

**Measured with `rhel8-b5399` on the 2-socket server**, in [measurements.md](https://github.com/muller-camille/ik_llama.cpp/blob/main/docs/deepseek/measurements.md): no other setting of the draft is faster than the one in use (30.4 tokens/s over five subjects), and the threads held up in the run of `rhel8-b5397` are not from the code of the experts.
