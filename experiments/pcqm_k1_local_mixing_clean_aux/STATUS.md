# K1 local mixing status

Desktop-owned branch: `codex/exp/k1-local-mixing-clean-aux`.

## Kaggle3 submitted pair

- Kernel: `nvoid912/molgap-k1-v4-ssma-pair-100k-s42-v1`, kernel ID 136612854, actual version1.
- Last authoritative observation: RUNNING, 2026-10-01; see
  [retained scheduler/source observation](kaggle3_submission_v1/remote-source-acceptance.json).
- Arms: original K1 V4 reference and bounded SSMA joint aggregation;
  each100K/40epoch/FP32, 31240steps and3998720 formal presentations.
- Frozen training source: `70e0b3af981df106de31eef5dcbdb3870eb74a7e`.
- [Training contract](training_protocol_kaggle3_v1.md),
  [Spec](experiment_spec_kaggle3_v2.json), [submission records](kaggle3_submission_v1/).
- Prospective trajectories: [reference](kaggle3_v2/reference/trajectory.json),
  [candidate](kaggle3_v2/ssma/trajectory.json). Local logical version2 names
  the repaired source binding; the actual first remote kernel submission is version1.
- Local CPU loading, strict initial tensor load, release inputs and focused
  synthetic regressions passed. Actual T4 runtime qualification is remote
  and precedes either formal training worker. No remote result acceptance yet.
- Two independent complete replay entries remain pending terminal artifacts,
  strict same-job comparison, actual costs and V5/RML qualification.

## Preserved earlier dispositions

The previous two-candidate qualification was NO_TRAIN because the historical
reference could not qualify for strict replay. The user then explicitly authorized
fresh reference training on Kaggle3. The first new source binding was closed
NO_TRAIN before any remote submission when the packager rejected its own shared
artifact-owner Python module. These are infrastructure dispositions, not module negatives.

## Reconciliation

Reconcile this exact kernel/version; do not resubmit from a stale local summary.
Retain minimal per-arm evidence on terminal state, accept reference before candidate,
write attribution and follow BRANCHES terminal routing. No automatic scale-up.
The post-submit adapter fix for `/code/owner/slug` responses is outside the frozen
remote source; the running payload is unchanged. Pulled entry bytes differ only
by LF/CRLF and both raw hashes are retained.

The imported historical K1 reference still lacks its local raw trace pointer for
portable repository checks; this is not claimed repaired by the fresh submission.
