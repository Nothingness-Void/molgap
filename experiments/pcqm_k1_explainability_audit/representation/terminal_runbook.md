# Representation diagnostic terminal workflow

Prepared 2026-09-30. Instructions only: no result or promotion is assumed.
The submitted source remains immutable. All commands run from the repo root.

## 1. Reconcile and retrieve

Use the owning SCNet workload skill and ignored `.scnet/kunshan.json` routing.
Read the exact job from `submission.json`; do not infer success from a monitor
message, absent queue entry, or partial output. Retain native `sacct` output
with job ID, state, exit code, elapsed allocation, start/end, CPU usage and
memory when available. Keep units explicit; missing CPU time stays unknown.

Download the attempt output and both logs into a new ignored local directory:

```text
platforms/_records/scnet/k1_representation_audit/attempt-123315282/
  raw/                         original output JSON files, unchanged
  logs/                        stdout/stderr, including failures
  sacct.txt                    original scheduler evidence
  scheduler.json               exact job_id/state/exit_code/elapsed_seconds
```

`scheduler.json` is a transcription of retained scheduler output, not a guess.
Successful acceptance requires state `COMPLETED`, exit code `0:0`, and job ID
matching the submission receipt. Never rerun or modify the remote job just
because acceptance is not ready. Keep partial output on any failure.

## 2. Independent saved-artifact acceptance

```powershell
.venv\Scripts\python.exe experiments/pcqm_k1_explainability_audit/accept_representation.py `
  --raw platforms/_records/scnet/k1_representation_audit/attempt-123315282/raw `
  --scheduler platforms/_records/scnet/k1_representation_audit/attempt-123315282/scheduler.json `
  --output platforms/_records/scnet/k1_representation_audit/attempt-123315282/accepted
```

This CLI reads JSON only. It does not construct a model, load torch tensors,
run inference, fetch data or write a scientific decision. It refuses to
overwrite a prior acceptance directory.

Required checks include exact source/job/checkpoint identity, 16 chunks of
128 rows, the predeclared 1,024-row panel in both scales, matching targets and
descriptors, all artifact hashes, protected-role flags, reproduction tolerances,
finite/bounded spectra and sensitivities, and declared role events.

Outputs:

- `acceptance.json`: mechanical verification, caveats and raw manifest digest.
- `analysis.json`: scale-specific panel MAE, paired representation changes,
  original input-visible strata, and linear atom-count-adjusted correlations
  between metric changes and absolute-error changes.
- `REVIEW_REQUIRED.md`: controller review reminder; not a terminal verdict.

The current producer asserts hooked/unhooked invariance, gradient isolation
and unchanged model state before successful completion, but does not retain
per-batch maxima. State this provenance limitation explicitly. Do not invent
unrecorded maxima, gradients or hidden states. No inference rerun is implicit.

## 3. Controller analysis

Read all three layers and report all predeclared metrics, including negative
and null results. Do not select only favorable layer/stratum combinations.
Groups can overlap; they are not additive population partitions. Very small
groups are descriptive only. The atom-count adjustment is linear and does
not control scaffold, chemistry, exposure, or seed effects.

Questions:

1. Does normalized node rank actually decrease across each exchange, or do
   retained local states remain diverse?
2. Does rank change across scales align with error changes on the *same* rows?
3. Does downstream prediction sensitivity change without rank degradation?
4. Are patterns coherent in original topology strata, rather than driven by
   molecule size or a few extreme rows?
5. Could differing exposure, checkpoint selection or random training paths
   explain the contrast? Two endpoints do not separate these explanations.

Decision options, none automatically released:

| Finding | Permitted interpretation | Follow-up boundary |
|---|---|---|
| No coherent rank/sensitivity/error association | No supported representation failure mechanism | Close this explanation; no rank-repair model |
| Rank changes but residuals do not align | Representation geometry changed, quality effect unknown | No architecture nomination from rank alone |
| Sensitivity/error relation without node collapse | Communication direction/addressing hypothesis | Check prior closed routes before proposing a distinct intervention |
| Coherent relation across strata | Diagnostic lead, not causal proof | Separate prospective single-mechanism plan and compute decision |
| Missing/bad output or failed runtime gate | Infrastructure or evidence failure | Preserve artifacts; no scientific negative verdict |

Do not compare the two panel MAEs as an architecture experiment: the same
architecture was trained on different scales/exposures. Do not call these
reused development rows an independent test or reopen prior closed routes.

## 4. RML closure (only after interpretation)

Use the existing `molgap.research_memory.terminal_wiring.close_terminal_arm`
and the shared terminal descriptor/finalization path. This is one NO_TRAIN
trajectory containing two checkpoint observations, not two training arms.

Required evidence to prepare:

- Immutable `rml_plan/trajectory.json` and its original decision state.
- Actual run binding from `submission.json`; the plan action had no scheduler
  ID before submission. Retain the receipt as the binding instead of editing
  the prospective snapshot to claim that it knew the future job ID.
- Hash-bound raw manifest, independent acceptance, analysis, scheduler evidence,
  dated controller decision and seven separate V5 outcomes.
- Observed prediction-input, labels-read and metric-computed roles from actual
  events. No training/selection/submission role event may be fabricated.
- Actual native allocation/process cost, keeping queue and CPU time unknown
  until scheduler evidence supplies them; the rejected preallocation request
  is operational history, not a fictitious accelerator run.

Do not directly reuse `k1_relation_audit_records.prepare_no_train_terminal`:
it assumes particular relation-audit output/role layouts and a prospective
action already containing the physical run ID. Its assumptions do not match
this diagnostic. Use the generic terminal path with the actual receipt.

No finalized terminal JSON is prefilled here: scientific outcome, real costs,
artifact hashes and terminal time do not yet exist. After finalization run:

```powershell
.venv\Scripts\python.exe -m molgap.research_memory validate
.venv\Scripts\python.exe -m molgap.research_memory rebuild
.venv\Scripts\python.exe -m molgap.research_memory check --frozen
```

Commit compact accepted evidence, the decision, terminal RML and changed
derived files on server. Leave large raw row records in ignored storage.
Close the exact monitor event through the existing controller; do not wake or
adopt desktop work. No training trace, strict causal win, training replay-ready
entry, full training, protected-role read or successor is implied by closure.
