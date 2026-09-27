# K1 relation resolution

One question: does preserving atom-pair information improve portable K1
regression? Read `protocol.md` for hypotheses and `training_contract.json`
for immutable settings. `STATUS.md` routes execution evidence; per-arm RML
plans live under `arms/`. Training and post100K NO_TRAIN audits are separate.

Reusable models/runtime/records live in `src/molgap/k1_relation_*`.

Decisions: [frozen portability](audit/decision.md) and
[frozen relation dependence](diagnostic/decision.md). Knockout dependence is
not a substitute for a matched from-scratch architecture benefit.
