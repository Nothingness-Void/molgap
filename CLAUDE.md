# CLAUDE.md

Entry point for Claude Code. The full reading protocol and constraints live in
**`AGENTS.md`** — read it first.

Fastest path to context:
1. `AGENTS.md` — reading protocol + hard constraints
2. `CURRENT_STATE.md` — what's true right now (recommended model, next actions)
3. `ARCHITECTURE.md` — which file to edit for a given change

One non-negotiable, repeated here because it's the most common mistake:
**always use `.venv\Scripts\python.exe`** (system Python lacks torch/pyg), and keep
training/inference on the same conformer method (ETKDG).
