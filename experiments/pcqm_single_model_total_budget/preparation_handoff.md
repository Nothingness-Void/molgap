# Preparation Handoff

Local preparation only. No remote access, publication, submission, commit or push
was performed by this worker. Parent-owned profile files and README were not edited.

## Changes

- experiment_catalog.py: two explicit GPTrans/2 shape addons.
- gptrans_bond_local.py: mode-to-approved-shape dispatch through the existing
  factory; default initialization unchanged; fresh norm reuses gptrans_variants.
- experiment_family_workflow.py: delivery limited to wide reference/deep candidate,
  exact approved addons, fixed loss/data/base/sampler/transform/overrides; only
  initialization state hash may differ, not kind/seed. Recipe fields other than
  mode/state hash must match. No comparison-readiness or promotion rule changed.
- experiment_input.py: pin delivery comparison source; permit first freeze over
  exactly the declared CPU .pt files, never unrelated/incomplete freeze files.
- tests/test_gptrans_pair_norm.py and tests/test_scientific_pair_input.py:
  strengthen existing paths, no new test file or framework.
- experiment.json, authority.json, protocol.md, decision.md, per-arm recipes/data
  declarations, initialization_manifest.json and candidate research-action policy.
  Policy promotion/early-stop rules remain null.

## Frozen Inputs

CPU scratch seed42 states are retained under
`D:/molgap-artifacts/runs/pcqm-single-model-total-budget-500k-s42/v1/inputs`.
prepare_inputs.py constructs, checks finite state, saves atomically, reopens with
the shared frozen-state inspector and loads through load_family_model. No data
payload was read. initialization_manifest.json owns file/tensor SHA and counts.
Both states match approved counts: A12478993, B7700481.

First release: 25200 seconds wall/14 T4-device-hours, 180s cleanup reserve.
Cumulative ceiling30 is separate; every continuation needs actual-cost accounting
and is limited to min(8,30-actual) device-hours and4h wall; no automatic trigger.
experiment.json timestamp was sampled with datetime.now().astimezone().isoformat(),
not guessed. No source commit or scheduler observation was invented.

## Verification

The first focused invocation ran both existing test files:22 passed,2 new delivery
fixture failures (old predictor guard was inherited). The fixture was repaired
without relaxing the guard. The full pair file then passed19, skipped1 K1-only
inapplicable delivery case. After the observed preseeded-input bootstrap repair,
only affected delivery/same-run nodes were rerun:5 passed,1 skipped. The shape
file's4 successful tests were not rerun unchanged. git diff --check passed.

Coverage includes default tensor SHA a65034da..., exact A/B CPU frozen loading,
only approved shape addons, seed invariance, rejection of optimizer changes,
unchanged strict architecture scope, actual terminal assessment non-strict with
architecture/initialization/runtime mismatch retained, no promotion decision
scope, and no PAIRED_ENDPOINT for incomplete terminal evidence. This is not GPU
native qualification, submission readiness, or a promised endpoint comparison.

The local submit_workflow entry initially blocked on missing input_binding.json
when only initial states existed. After the narrow bootstrap repair, direct
resolution passed metadata/state/scope checks and stopped at:
`ValueError: Runtime distribution has untracked record/source inputs`.
Formal preparation therefore remains pending parent review and commit. No
generated prospective trajectory was published; that is required before training.
Strict terminal replay qualification intentionally remains blocked for delivery;
do not use this noncausal path to obtain strict replay or model promotion.

## Parent Commands

After reviewing and committing all intended runtime/human inputs, from this
worktree run the existing local-only entry (no credentials, no --execute):

```powershell
& 'D:/文档/molgap/.venv/Scripts/python.exe' platforms/local/run.py --profile D:/molgap-machine-profile.json --script platforms/kaggle/submit_workflow.py -- --experiment experiments/pcqm_single_model_total_budget/experiment.json --repo-root .
```

CPU initialization/metadata generation, already completed, uses the primary
interpreter with this checkout selected, not an editable reinstall:

```powershell
& 'D:/文档/molgap/.venv/Scripts/python.exe' -c "import sys,runpy; sys.path.insert(0,'src'); runpy.run_path('experiments/pcqm_single_model_total_budget/prepare_inputs.py',run_name='__main__')"
```

prepare_inputs.py refuses conflicting immutable outputs. Do not clear a journal
or alter frozen state merely to make preparation pass. Parent owns the single
submission-owner quota/release recheck and all remote operations after review.
