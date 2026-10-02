# K1 local mixing status

Desktop-owned branch: `codex/exp/k1-local-mixing-clean-aux`.

## Kaggle3 accepted accuracy terminal

- Exact kernel `nvoid912/molgap-k1-v4-ssma-accuracy-100k-s42-v1`, ID136643751,
  actual version1 is authenticated COMPLETE. Entry matches source `f21920ba`;
  [observation](kaggle3_accuracy_reconciliation_v1/remote_observation.json).
- Original reference and bounded SSMA each completed100K/40epochs/FP32,
  31240steps and3998720 presentations; both selected epoch40. Independently
  accepted manifest-bound predictions, selected/resume states and traces retained.
- SSMA is **NEGATIVE_UNDER_CONTRACT**: MAE0.143361512 versus reference0.141294461eV;
  gain−2.067051meV, paired95% row interval[−3.005820,−1.105339]meV.
  [Decision](kaggle3_accuracy_reconciliation_v1/terminal_decision.md) and
  [attribution](kaggle3_accuracy_reconciliation_v1/attribution.md) own interpretation.
- The48.977% synchronized optimizer-step overhead is reported under the
  explicitly authorized report_only policy. Other execution gates passed.
  [Frozen protocol](training_protocol_kaggle3_accuracy_v1.md) is unchanged.
- Independent reference/SSMA terminal RML is finalized; strict replay/readiness
  admission remains unevaluated. No retry,successor,scale-up or adoption.
  Prior cost-gated NO_TRAIN records retain their original authority.

## Previous Kaggle3 cost-gated terminal pair

- Kernel: `nvoid912/molgap-k1-v4-ssma-pair-100k-s42-v1`, kernel ID 136612854, actual version1.
- Authoritative terminal state: ERROR, 2026-10-01; see
  [scheduler observation](kaggle3_submission_v1/remote-status-final.json).
- Accepted disposition: both arms NO_TRAIN. Reference runtime qualified;
  SSMA optimizer-step overhead28.563243% exceeds the frozen25% maximum.
  The pair barrier prevented both formal workers from starting.
- [Attribution](attribution_kaggle3_v1.md); canonical terminal evidence:
  [reference](kaggle3_v2/reference/rml_finalized/v5_evidence.json),
  [candidate](kaggle3_v2/ssma/rml_finalized/v5_evidence.json).
- Arms: original K1 V4 reference and bounded SSMA joint aggregation;
  planned each100K/40epoch/FP32, 31240steps and3998720 formal presentations.
  Actual formal epochs/steps/presentations: zero for both.
- Frozen training source: `70e0b3af981df106de31eef5dcbdb3870eb74a7e`.
- [Training contract](training_protocol_kaggle3_v1.md),
  [Spec](experiment_spec_kaggle3_v2.json), [submission records](kaggle3_submission_v1/).
- Prospective trajectories: [reference](kaggle3_v2/reference/trajectory.json),
  [candidate](kaggle3_v2/ssma/trajectory.json). Local logical version2 names
  the repaired source binding; the actual first remote kernel submission is version1.
- Local CPU loading, strict initial tensor load, release inputs and focused
  synthetic regressions passed. Actual T4 runtime qualification is remote
  and blocked formal training at the frozen cost gate.
- RML terminal closure/rebuild passed for both NO_TRAIN records. No training
  replay entries were added; no accuracy result or model promotion exists.

## Preserved earlier dispositions

The previous two-candidate qualification was NO_TRAIN because the historical
reference could not qualify for strict replay. The user then explicitly authorized
fresh reference training on Kaggle3. The first new source binding was closed
NO_TRAIN before any remote submission when the packager rejected its own shared
artifact-owner Python module. These are infrastructure dispositions, not module negatives.

## Reconciliation

This exact kernel/version is terminal; do not resubmit from a stale local summary.
Only minimal logs/runtime/cost JSON were retrieved. Complete training outputs do
not exist. The experiment branch remains retained under BRANCHES while reusable
infrastructure integration and final Git routing await review. No model adoption
or successor is authorized by this cost rejection.
The post-submit adapter fix for `/code/owner/slug` responses is outside the frozen
remote source; the running payload is unchanged. Pulled entry bytes differ only
by LF/CRLF and both raw hashes are retained.

The imported historical K1 reference still lacks its local raw trace pointer for
portable repository checks; this is not claimed repaired by the fresh submission.
