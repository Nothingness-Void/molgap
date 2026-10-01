# Same-reference Replay qualification

On 2026-10-01, the two accepted G1/G2 transactions were requalified against the
additive [control qualification](../../../pcqm_gptrans_v5_audit_reference/results/reference_qualification/decision.md).
The original plans, prospective snapshots, source package, predictions,
checkpoints, terminal outcomes and finalization receipts were preserved.

## What changed

The original bundle, its file SHA and semantic digest were frozen before each
arm. The verified reference qualification proves that the replacement bundle
names the same control, dataset, transform, recipe, observations, checkpoint
and predictions. Only role/cost/acceptance representations and enrollment
metadata changed. Candidate qualification reverified all terminal bindings and
recomputed the existing strict assessor with those distinct reference pointers.
It did not substitute a stronger baseline or add future information to a plan.

The candidate manifests had inherited the control's `model_identity`; the
sidecars correct it to each candidate's independently accepted architecture
fingerprint, already recorded in its comparability identity. Other manifest
fields are unchanged except verified eligibility. No model was changed.

## Verified admission

| Trajectory | Effective comparison | Trace capability |
|---|---|---|
| `TC-gptrans-author-degree-scale-100k-s42` | STRICT_CAUSAL, declared degree initialization intervention only | complete |
| `TC-gptrans-author-path-bond-mean-100k-s42` | STRICT_CAUSAL, declared path input intervention only | complete |
| `TC-gptrans-v5-audit-reference-100k` | accepted reference, no winner label | complete |

All three entered one actual Replay group with 60 observations, 46,860 optimizer
steps and 5,998,080 sample presentations. Per-arm `rml_plan/candidate_qualification.json`
binds the old receipt, prelaunch, acceptance, reference migration and published
qualified manifest/comparison records. RML validation and Replay construction
recompute these proofs rather than trusting status strings.

The original paired-endpoint records remain unchanged historical records.
The effective STRICT_CAUSAL qualification is a terminal metadata reassessment,
not a retroactively improved release audit. Original gains and the frozen
0.003 eV gate were not changed. Single-seed and EMA convergence caveats remain.

Git newline conversion differed from retained, accepted bytes for five bound
inputs: screen configuration, submission receipt, target-transform asset, row
manifest and target manifest. Those exact bytes were preserved in Git using
narrow byte-preservation attributes. No scientific content or bound digest was
rewritten to fit converted files.

Replay actual cost excludes prospective budget estimates from incurred sums,
but retains them as separate planned measurements. The two arms' measured
allocation totaled 7.1523777060 T4 device-hours; the six control segments totaled
5.0155572684. Unknown CPU/queue measurements remain unknown. No costs or role
events were added; source ledger events were preserved.

## Boundary

No training, inference, protected-role access, remote submission, multi-seed,
scale bridge or full-training release occurred. Future training screens must
verify closure feasibility before submission and actual pair admission after
terminal acceptance, as specified in the [standard workflow](../../../../docs/operations/EXPERIMENT_WORKFLOW.md#replay-first-release-and-closure).
