# Closure note (2026-11-23)

- The RML policy, plan and trajectory were published after the tail was scored, not before. The frozen record that predates the result is `protocol.md` (commits fc6cdc2e7, 005c2ca13). The trajectory binds that commit as its source and `protocol.md` as its contract; it adds no rule.
- Scoring ran once, through `score_c4_once.py`, which stops if its output exists. No other tail number was computed for C4.
- Closure: policy `pcqm-gptrans-full-wd-c4`, `plan`, `terminal-pipeline` (outcome NEGATIVE_UNDER_CONTRACT, pipeline COMPLETE, next decision SOL_REQUIRED), then `check --frozen` passed. No trace was recorded (single-run closure, context-only comparison).
- Supporting prior evidence bound in the plan: the four 500K weight-decay records (s43, s44; wd_low, wd_high).
- Leaderboard: `gptrans_c4_wd022_s43` added with `tail_leaderboard_add.py`; board sha 98aa8c17...ef541.
