# Shared Checkpoint Efficiency

Execution-only K1/GPTrans finite-validation optimization. No molecular inputs,
training, inference, quality claim, sampling change or checkpoint-policy change.

- [Protocol](protocol.md): scope and bounded diagnostic.
- [Execution audit](audit.md): reused owners, fixed overhead and unsafe shortcuts.
- [Prospective decision](decision.md): immutable planned state.
- `prepare.py`: existing RML planner; freezes code/config before native timing.
- `run.py`: thin wrapper over `molgap.checkpoint_diagnostic`.
- `close.py`: metadata acceptance and existing V5/RML finalizer.
- `results/`: matched original/batched checker and atomic-save measurements.
- `terminal_decision.md` / `attribution.md`: accepted result and limitations.

Original checker exists only in `legacy_oracle.py` as a frozen test oracle.
It is not a training option, fallback, addon or dispatch entry.
