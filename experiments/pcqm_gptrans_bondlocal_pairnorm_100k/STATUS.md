# Status

## Terminal acceptance: 2026-10-27 JST

Kaggle3 kernel `140232816` / version 1 is `COMPLETE`. `bond_local` is `CLOSED`; `pair_update_norm` is `POSITIVE_UNDER_CONTRACT`. Strict validation is `VERIFIED`, the candidate comparison is `STRICT_CAUSAL`, and both complete replay entries have no exclusions. See [qualification](qualification.md) and [terminal decision](terminal_acceptance/decision.md).

Strict closure report: `D:/molgap-artifacts/retrieved/pcqm-gptrans-bondlocal-pairnorm-100k-s42/v1/strict_closure_final.json`.


## Observation: 2026-10-23 JST

Kaggle3 accepted kernel `140232816/version1`:
[exact job](https://www.kaggle.com/code/nvoid912/molgap-gptrans-bondlocal-pairnorm-100k-s42).
Scheduler observed RUNNING at `2026-10-22T15:17:55Z`. Startup transport ended
with ConnectionError after a package-install log; native qualification and
formal training progress are not yet established. No duplicate submission.

Owner: `codex/exp/gptrans-bondlocal-pairnorm-100k`; frozen source
`32b96f619404160c9c685ba9bd8d1d79d77921f2`. The [protocol](protocol.md)
owns arms, gates and budget. No K1 job, monitor, successor or model adoption.

[Submission record](submission/record.json) retains physical/source identities
and original observations; [launch receipt](submission/launch_receipt.json)
binds both arms. Canonical prospective records are under
`runs/pcqm-gptrans-bondlocal-pairnorm-100k-s42/prospective/`.
Local run custody: `D:/molgap-artifacts/runs/pcqm-gptrans-bondlocal-pairnorm-100k-s42/v1/`.

## Executed checks and preparation incident

One CPU-only selection passed 22 tests in 100.05 seconds: existing pair-norm
model tests, bootstrap budget nodes and the scientific-input preparation journey.
It covered real combined-model construction, prepared forward/backward,
one optimizer update, in-memory weight roundtrip and config budget propagation.
This is not native T4 qualification or a completed accuracy/resume comparison.

The release input check passed. Initial prospective publication completed both
arms, but global RML rebuild encountered two inherited, ignored checkpoint paths
under `pcqm_gptrans_local_inductive_bias_100k_s42_v2/attempt_002/arms/`.
Their existing primary-checkout keepers were SHA256-verified and hardlinked
(`rwse16`: 21089307 bytes; `rwse16_local_edge`: 23910393 bytes), not downloaded,
loaded or retrained. No independent backup or extra payload copy is claimed.
RML rebuild and `check --frozen` then succeeded (not a portable/replay claim).
Reconciliation validated the published identities
and used existing freeze/bind owners to finish the original preparation,
retaining the initial failure in its report. No gate or evidence was bypassed.
