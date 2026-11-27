since: rhel8-b5399

**The saved conversations no longer belong to one context size.**

- Until now a file of the prompt cache (`--cache-ram-dir`, `CACHE_DIR`) was taken back only by a server with the context size that wrote it: after a change of `CTX` the conversations stayed in the directory, unused.
- Now the server takes back every conversation that fits in its context, whatever the context it was written with. The files written by `rhel8-b5390` to `rhel8-b5399` are found too.
- A conversation that is longer than the new context is left in the directory, for a server with a larger one. The start of the server says how many there are of each kind.
- If a file does not come back in a server that was set up in another way (context, batch size, number of slots), it is left where it is instead of being removed, and the prompt is read again.
- Checked without a model: `test-prompt-store` (the files of an earlier fingerprint are found, read and removed like the others). The way through the server with the model is not tested: the first restart with another `CTX` is that test, and the log says whether each conversation came back.

Nothing survives a reboot of the machine, as before: `/mnt/dscache` is a RAM disk.

**Measured with `rhel8-b5399` on the 2-socket server**, in [measurements.md](https://github.com/muller-camille/ik_llama.cpp/blob/main/docs/deepseek/measurements.md): no other setting of the draft is faster than the one in use (30.4 tokens/s over five subjects), and the threads held up in the run of `rhel8-b5397` are not from the code of the experts.
