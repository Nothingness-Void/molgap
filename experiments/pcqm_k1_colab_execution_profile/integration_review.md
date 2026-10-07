# Non-promotion integration review — 2026-10-08

Reviewed question branch: `codex/exp/k1-colab-profile-a100`, based on verified
`molgap-desktop` a4236f5caf804bb7e8f514574ac12e184385a4d3. Independent bounded
A100 diagnostic; Kaggle500K source/job was not changed.

Reuse: accepted K1 frozen loader/factory/forward/objective, shared deterministic
settings and atomic IO/hash, RML plan/finalize/rebuild/check, package-only release
import bootstrap. Added: bounded execution profiler, thin Colab adapter and
experiment-owned acceptance translation. No replacement trainer or RML framework.

Focused final check: tests/test_k1_execution_profile.py,4 passed, one PyG
DeprecationWarning. The actual serialized WedgeData symbol/import origins,
executed payload binding and12 finalized artifact SHA256 values were verified.
RML rebuild and check --frozen passed. Raw remote results remain unchanged.

Accounting distinction: worker body58.803s excludes imports; cost events use
observed whole-process63.702s. Parent CPU63.527s is the body process CPU counter,
excluding imports and child processes. Full session CCU/busy time is unknown.
The role_use summary names prediction_input; detailed observed train_probe role
events also record labels_read, so this is not an exclusive access declaration.

Route under BRANCHES: accepted diagnostic with reusable implementation to desktop,
explicitly without model adoption, training release or full promotion. Frozen
source/sample package stays in the retained owner staging and private Drive.
