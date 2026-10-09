# K1 T4 cost-quality preparation

[Protocol](protocol.md) owns the frozen question and gates;
[evidence review](evidence_review.md) owns rationale. `profile/` and shared
trainer/runtime are parent/other-agent owned. This wrapper adds no model,
training loop, launcher, monitor or acceptance framework.

From the owner checkout with its configured project virtual environment:

```powershell
.\.venv\Scripts\python.exe experiments/pcqm_k1_t4_cost_quality/prepare_training.py
.\.venv\Scripts\python.exe experiments/pcqm_k1_t4_cost_quality/prepare_training.py --stage-local experiments/pcqm_k1_t4_cost_quality/draft-v1
.\.venv\Scripts\python.exe -m pytest tests/test_k1_cost_quality_preparation.py tests/test_same_run_replay.py -q
```

Validation checks the real retained initialization without model construction,
inference or data-role consumption. Staging writes a fresh unpublished draft
set: Spec, two recipes/plans, policy, role plan, workflow/acceptance drafts and
blocker report. Draft paths inside Spec intentionally describe eventual
question-root files; the nested draft directory is NOT an executable release.
Never overwrite a published record. No stage calls `plan-prospective`,
`prepare-workflow`, remote APIs or RML rebuild.

Parent must review/install draft sidecars at their declared question-root paths,
register the explicit policy via the RML owner, validate the parent-owned shared
same-run acceptance/target-identity support, commit executable source, and
regenerate bindings. The acceptance format pins contract and original target
manifest only; Spec prospective bindings own the future mean2 reference.
It fabricates no prelaunch comparison or accepted historical mean2. Only after
profile/local diagnostic and allocation-timeout/durability gates pass may parent
use the existing CLI:

```powershell
.\.venv\Scripts\python.exe -m molgap.experiment_cli validate-spec --spec experiments/pcqm_k1_t4_cost_quality/experiment_spec.json
.\.venv\Scripts\python.exe -m molgap.experiment_cli prepare-workflow --spec experiments/pcqm_k1_t4_cost_quality/experiment_spec.json --repo-root . --plan experiments/pcqm_k1_t4_cost_quality/workflow_plan.json --output experiments/pcqm_k1_t4_cost_quality/prepared-v1
```

These are conditional parent commands, not commands executed by this task.
`prepare-workflow` publishes canonical prospective records only after shared
static gates; local drafts alone do not meet the before-training trajectory
requirement. Parent handles Kaggle release, actual scheduler identity and
retained output acceptance through the platform skill and `accept-workflow`.
Strict readiness and the future mean2 accepted reference remain pending.

## Parent release API gap

No `--parent-release-file` or release-enabled `build_inputs` parameter is exposed
yet. The profiler produces `result.json` (`molgap-k1-native-t4-profile-v1`,
`status=complete`) and hash-bound `completion.json`, but neither is a parent
acceptance decision. No clean-fit decision schema/validator is available.
An accepted boolean, completed worker or unbound timing ratio cannot release TRAIN.

The release owner must provide repository-local `{path, sha256}` bindings for
the profile acceptance and completed local clean-fit decision, plus a pinned
release record. Reuse `evidence_pointers.resolve_repo_pointer` and
`verify_bound_artifact` for confinement/hash checks. The owning acceptance
validator must verify actual native T4 hardware/run/source/payload identity,
complete retained timing artifacts, matched single/mean2 step savings >=0.25,
and the completed clean-fit decision's absence of an urgent fitting failure.
It must preserve the pair ceilings8 allocated T4 device-hours/14400 wall seconds,
and distinguish accepted evidence from raw completion and planning estimates.

Until that API is frozen, parent release/publication is separate from these
prep-only plans. Do not mutate a published plan or trajectory into TRAIN. Before
publication, parent must freeze a distinct release-bound plan with chosen action
`TRAIN_PAIR_100K`, action type `paired100k_after_parentrelease`, next allowed
action exactly `A001` (the bounded pair only), and the release record in
`contract_refs`. No automatic500K/full or remote action follows. This script
does not synthesize those decisions or duplicate the acceptance validator.
