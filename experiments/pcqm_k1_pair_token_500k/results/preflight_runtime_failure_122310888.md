# Preflight runtime failure 122310888

Kunshan job `122310888` reached the candidate preflight and exited before its
first optimizer step because the platform DTK PyTorch does not expose the
newer `AdamW(..., fused=False)` keyword. The platform implementation is
unfused by default. Removing only that unsupported explicit keyword preserves
the frozen unfused, `foreach=False` optimizer semantics. No training epoch or
scientific result was produced.

