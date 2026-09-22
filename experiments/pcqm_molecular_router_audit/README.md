# PCQM molecular specialist audit

This experiment asks whether already-completed PCQM models contain useful,
input-identifiable specialist behavior. It does not train a Router or a model.

- `protocol.md` freezes scope and gates before the audit is run.
- `run_audit.py` is the thin CLI.
- `decision.md` records the terminal scientific decision and points to compact
  machine evidence in `results/decision.json` and the other `results/*.json`
  reports.
- The aligned row-level prediction matrix remains under ignored platform record
  storage; only its identity and SHA-256 enter Git.
