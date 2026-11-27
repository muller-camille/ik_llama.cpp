since: rhel8-b5397

**The routed experts are computed as before `rhel8-b5397` again.** The run of `rhel8-b5397` on the 2-socket server measured what it changed:

| Node, for one token | `rhel8-b5390` | `rhel8-b5397` | Threads busy |
|---|---|---|---|
| Shared expert (`ffn_up_shexp`) | 3.09 ms | 2.91 ms | 83% to 93% |
| Routed experts, up and gate, then down | 13.95 ms | 13.83 ms | 91-93% to 96-97% |
| The whole graph | 45.46 ms | 45.48 ms | |

- In the shared expert the threads take rows from each other and the node is shorter: that stays.
- In the routed experts the threads were busier and the nodes no shorter (the down matrices were slower): the memory sets the speed there. That part is taken out; these nodes are the code of `rhel8-b5390` again.
- Tokens/s did not move beyond what two runs differ by (21.73 and 21.73 after the long prompt, 22.78 and 23.11 after the short one), and the texts are the same. The estimate of up to 1 ms per token was too high.

**`profil.sh` compares settings of the draft.** The drafter stops a draft at the first token it is less than 75% sure of, a default that was never compared with others on this server. `SPEC=dspark:n_max=4,p_min=0.5` among `VARS` is a pass with that setting, with the duration of a draft step and the tokens it gives in the final table; `SANS_DRAFTER=0` leaves out the passes without the draft.

The measurements are in [measurements.md](https://github.com/muller-camille/ik_llama.cpp/blob/main/docs/deepseek/measurements.md).
