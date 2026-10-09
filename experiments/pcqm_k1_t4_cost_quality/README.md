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

## Optional Parent Release

`build_inputs(..., parent_release_file=Path(...))` and `--parent-release-file`
validate an explicit human-controller release before building fresh unpublished
plans. Omit the option to retain PREPARE_ONLY behavior. No option publishes,
trains, submits or overwrites a record. Existing question-root training plans or
prospective output directories block release-enabled preparation.

The parent must author the following repository-local record inside this owning
question, outside `profile/`. Every evidence pointer has exactly `path` and
`sha256` (actual file-byte SHA256, not an accepted boolean). Placeholder hashes
below are not a release:

```json
{
  "format": "molgap-k1-t4-parent-release-v1",
  "controller": "human-controller",
  "approved_by": "human controller identity",
  "approved_at": "explicit approval timestamp",
  "source_commit": "exact source commit used for preparation",
  "action": "TRAIN_PAIR_100K",
  "allowed_actions": ["TRAIN_PAIR_100K"],
  "budget": {"allocated_t4_device_hours": 8, "wall_seconds": 14400},
  "profile_acceptance": {
    "path": "experiments/pcqm_k1_t4_cost_quality/profile/acceptance.json",
    "sha256": "actual SHA256"
  },
  "clean_fit": {
    "terminal": {"path": "experiments/pcqm_k1_clean_fit_generalization_500k/terminal.json", "sha256": "actual SHA256"},
    "acceptance": {"path": "experiments/pcqm_k1_clean_fit_generalization_500k/acceptance.json", "sha256": "actual SHA256"},
    "decision": {"path": "experiments/pcqm_k1_clean_fit_generalization_500k/terminal_decision.md", "sha256": "actual SHA256"},
    "finalization": {"path": "experiments/pcqm_k1_clean_fit_generalization_500k/closure_receipt.json", "sha256": "actual SHA256"},
    "assessment": "NO_URGENT_FITTING_FAILURE",
    "rationale": "Human assessment explaining why the pinned bounded NO_TRAIN findings require no urgent fit repair before this pair; preserve scientific limits."
  }
}
```

`validate_parent_release(root, release_file, source_commit=...)` reuses
`resolve_repo_pointer` / `verify_bound_artifact`. It calls the parent-coordinated
read-only `profile.close.accept(root)` only after that callable exists. This
owner checks actual retained native T4 hardware/run/source/payload/timing bytes;
the pinned acceptance's analysis must equal the recomputed owner analysis.
`analysis.single_step_saving_fraction` must be finite and >=0.25. No copied
native-profile validator, raw completion or `accepted=true` grants authority.

Clean-fit bindings must name the canonical files integrated into this same
repository. Its `molgap-rml-finalization-v1` receipt must finalize NO_TRAIN,
bind `terminal_input.json` to the pinned terminal object's shared `json_bytes`
serialization (the terminal file's raw bytes are independently SHA-pinned), and pin the acceptance
and decision in `input_artifact_hashes`; terminal `artifact_hashes` must agree.
Acceptance must retain complete/no-training, hash-verified, NO_TRAIN outcomes.
The manual assessment is parent judgement, not a new scientific inference
validator or causal underfit/overfit claim. The clean-fit NO_TRAIN disposition
itself does not authorize this separate pair.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
.\.venv\Scripts\python.exe experiments/pcqm_k1_t4_cost_quality/prepare_training.py --parent-release-file experiments/pcqm_k1_t4_cost_quality/parent_release.json --stage-local experiments/pcqm_k1_t4_cost_quality/release-draft-v1
```

The release file is SHA-bound in each fresh plan's `parent_release` and included
in `contract_refs` before publication. Chosen action is `TRAIN_PAIR_100K`, action
type is `paired100k_after_parentrelease`, next allowed action is exactly `A001`.
Policy approval/budget pointers and prospective plan hashes/Spec identity are
rebound together. Staging remains unpublished and `training_authorized=false`:
parent registration, publication, executable-source and runtime/durability gates
remain separate. Neither500K/full nor remote actions follow automatically.
Never convert an already published PREPARE_ONLY trajectory into TRAIN.

The active CLI root is `molgap.constants.REPO_ROOT`; explicit `root` callable
arguments remain supported. The shared RML planner permits extra input fields
but does not publish top-level `parent_release`. Its durable release binding is
the contract pointer hashed into canonical `decision_state.source_hashes`.
Preparation tests exercise this through read-only `_prepare_plan`, not publication.
Human approval identity/authority must reflect the actual user instruction and
parent assessment; the wrapper creates neither an approval record nor an identity.
