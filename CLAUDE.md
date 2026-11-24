# Rules for Claude in this repository

## No local experiments

Set by the owner of this fork on 2026-11-24, after having to say it four times. It holds for every session.

- Do not run experiments in the session's own container: no benchmarks, no timing runs, no runs of a model or of
  `llama-server`, no profiling, no trial runs of the scripts in `scripts/deepseek/`, no stand-in models.
- When a measurement or a test is needed, say what it would answer, give the command to run (the shell on the
  server is csh/tcsh), and stop there. The owner runs it on the server and sends the result back.
- Work from the results the owner sends, as they were measured. Do not set a measured figure aside, and do not
  put an estimate or a figure from the container in its place.

Building the code to check that it compiles before a push is not an experiment.
