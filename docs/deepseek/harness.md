# DeepSeek Harness on the server

[README](../../README.md) · [changes](changes.md) · [releases](releases.md) · [measurements](measurements.md) · [setup](setup.md)

[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) is DeepSeek's open-source agent: a web UI, a command line and a headless mode around a model that reads and edits files and runs commands. Its own adapter for DeepSeek's API speaks the Messages protocol, which the server has at `/v1/messages`, and its model `deepseek-flash` is DeepSeek-V4.1 Flash. `harness.sh` starts it with the server of `deepseek.sh` in the place of that API:

```
npm install -g @deepseek-ai/dsh        # once; needs Node.js 24.9 or newer
./harness.sh                           # web UI on 127.0.0.1:3080, the address to open is printed
./harness.sh headless "run the tests"  # one task, the answer on the output
```

The script reads `API_KEY`, `HOST`, `PORT` and `CTX` from `~/.deepseek.conf` like the others (`SERVEUR=http://host:8080` for a server on another machine) and gives the harness an overlay (`dsh --patch`) that is rewritten at each start: the address of the server, the size of its context, so that the harness summarizes the conversation before the server runs out of it, and the reasoning level (`EFFORT=off|low|high|max`, also chosen in the UI). Nothing is written into the profiles of the harness, and the other providers set there stay as they are. The key is passed in the environment, not in a file.

- The agent runs, on the machine and with the account that start it, the commands the model decides to run. Start it in a work folder, with an account that has only the rights it should have; the script refuses to run as root unless told to (`EN_ROOT=1`, for a container).
- The web UI listens on this machine only. From another machine: `ssh -L 3080:127.0.0.1:3080 user@server`, then the printed address. `H_HOST` binds another address; over plain HTTP the address, which carries the access token, then travels in clear.
- The harness has telemetry towards DeepSeek, by default with the feedback a user submits; the script turns it off (`DSH_TELEMETRY_DISABLED`), and `TELEMETRIE=1` leaves it as the harness sets it.
- The server has one slot: the agents the harness can run side by side take turns on it.

Checked here with version 0.2.2-rc.5 of the harness and a small model in place of DeepSeek-V4.1: the web UI and the headless mode, a tool call and its result through the server and back, with and without internet access. The release workflow can repeat the headless run on a RHEL 8.10 image (`test_harness`); it did for `rhel8-b5352`, with Node.js 24.22.3 from nodejs.org.
