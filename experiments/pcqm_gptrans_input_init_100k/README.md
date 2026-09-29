# GPTrans-T input-embedding initialization screen

This desktop-owned experiment asks whether changing only the initial scale of
GPTrans-T's input embedding tables improves the fixed 100K/50K PCQM4Mv2 Gap
screen without adding parameters or forward operations.

- `protocol.md` owns the scientific question, comparison limits, and decision
  gates.
- `training_contract.json` owns the machine-readable intervention, V4 recipe
  identity, resource ceiling, and acceptance thresholds.
- `launch_decision.md` records why this one candidate screen was selected.
- The accepted reference and its immutable artifacts are owned by
  `../pcqm_gptrans_t_100k_v4/`; they are reused, not retrained here.
- The prospective trajectory, once published through the experiment CLI, owns
  this experiment's action, cost, role, and terminal evidence pointers.
- `prepare_initial_identity.py` verifies the original V4 state artifact and
  derives the candidate's complete state hash without constructing a model.
- `run_candidate.py` is the thin Kaggle entry point. Its preflight and training
  stages use separate interpreters to preserve the V4 runtime fingerprint.

The local work in this checkout does not constitute a GPU preflight, submitted
run, accepted result, or permission to use an evaluation role. Before any
future remote action, publish the canonical prospective record and verify the
source package, Kaggle owner, mounted dataset, and native resource budget.

The intervention is an initialization-scale test of this adapted GPTrans-T
implementation. It is not a claim that the entire implementation matches the
paper's atom or bond input semantics.
