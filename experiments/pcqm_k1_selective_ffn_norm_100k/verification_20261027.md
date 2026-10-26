# Local terminal verification - 2026-10-27

Scope: saved artifacts and terminal closure only. No training, model forward,
remote operation, protected-role access or model adoption occurred.

## Reused Owners And Repair

Reused `RunContext`, `TargetIdentityBinding`, `inspect_terminal_output`, terminal
staging/descriptor translation and `close_terminal_multi_arm`. The latter owns
canonical finalization and one consolidated RML rebuild.

The small shared repair resolves a local logical attempt from the original
Spec's SHA-pinned plan, after verifying the original package's Spec identity.
It does not replace the observed platform version with the attempt number.
Plans generated after source freeze may live outside the source archive, but
must match the original packaged Spec's digest. Wrong plan hashes, logical runs
and Spec identities remain rejected. Other platform identity rules are unchanged.

The original target-encoding recovery remains post-run and target-only; it is
not strict scientific admission. Source/config/output bytes were not rewritten.

## Executed Verification

- CPU-isolated `pytest -q tests/test_experiment_terminal_staging.py`:
  **19 passed**, including the real failure shape (unbundled but Spec-pinned
  local plan; launch version 1, frozen attempt v2 and producer version null).
- CPU-isolated `python experiments/pcqm_k1_selective_ffn_norm_100k/close_records.py`:
  six pipelines `COMPLETE`, all `validation_status=VALID`. V1 has no formal
  training traces; all three v2 trace statuses are `available`.
- `python -m molgap.research_memory --repo-root . check --frozen`: current.
- `git diff --check`: passed. No portable-clone or native hardware execution
  is claimed by these CPU checks.

The closure wrapper retains only this question's compact handoffs, pointing to
the existing global policy report rather than copying it into every arm.
Unaccepted metadata drafts from closure repair, including an interrupted
metadata publication, were quarantined under
`D:/molgap-artifacts/research/pcqm-k1-selective-ffn-norm-100k/rejected_closure_metadata_20261027/`.
No committed evidence or training artifact was deleted; the original accepted
v2 decision digest was restored and verified before completing all six pipelines.

## Replay Limit

The [receipt](terminal_checks/rml_closure.json) records zero new replay entries.
All three trained manifests explicitly exclude strict replay because the
reference and trace plan were not frozen before execution. Complete artifact
retention and native repeat/resume proofs do not supply those missing facts.
Do not rerun training just to repair historical bookkeeping.

The conservative synthesized manifests also list missing comparability-identity
fields. Their underlying dataset/split, optimizer/schedule, target transform,
precision and EMA settings exist in the frozen Specs and recipes; the missing
item is their qualified comparison binding, not erased training configuration.
Do not equate equal unknown placeholders with a validated strict comparison.

The [terminal decision](closure_decision.md#next-decision) owns the next proposed
discriminator: review the already-pending 500K initialization/forward question;
if two-model serving is desired, seek authorization for one independent BN seed
as a matched-cost ensemble control, not another normalization sweep.
