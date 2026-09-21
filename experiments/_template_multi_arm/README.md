# Multi-Arm Experiment Template — Native RML Terminal Closure

This directory provides the standard, reusable template for future multi-arm
experiments under the MolGap V5 Contract.

Future experiments do **not** write experiment-specific terminal wiring or custom
`prepare_rml_terminal.py` scripts. All candidate arms reuse the shared Desktop
infrastructure in `src/molgap/research_memory/native.py`.

---

## Invariants & Principles

1. **Independent Prospective Identities**:
   Multiple candidate arms may share a single physical Kaggle / remote GPU job,
   but **each candidate arm must have its own independently frozen identity**:
   - `trajectory_id`
   - `run_id`
   - `action_id`
   - `evidence_id`
   - `reference_id` + `reference_bundle_ref`
   If any arm in a multi-arm job lacks an independent prospective identity prior
   to launch, the prelaunch guard fails closed.

2. **No Post-Hoc Trajectory Creation**:
   It is strictly forbidden to create or patch a prospective trajectory after
   training starts or finishes. The prelaunch receipt must be committed to git
   in `source_commit`.

3. **Pure Machine Closure**:
   The terminal closure entrypoint automatically executes the full chain:
   ```text
   raw arm trace
       -> canonical_trace.json
       -> trace_manifest_terminal.json
       -> comparison_readiness.json
       -> terminal_evidence.json (v5_evidence)
       -> observed cost_records.json & role_history.json
       -> terminal.json package
       -> terminal-pipeline (atomic finalize -> validate -> rebuild -> check --frozen)
       -> replay admission
   ```

4. **Machine-Readable Replay Status**:
   Every arm deterministically yields `closure_status.json` containing:
   - `replay_status: "REPLAY_READY"` (candidate admitted to `replay_pool.json`)
   or
   - `replay_status: "REPLAY_EXCLUDED"` + explicit `reasons`

5. **Decision Authority**:
   Desktop `v5_desktop` authority and protected evaluation roles remain strictly
   controlled. Terminal closure always terminates with `next_decision: "SOL_REQUIRED"`.

---

## Experiment Lifecycle & Step-by-Step Guide

### Step 1: Copy Template & Name Experiment Directory
Copy this template directory to your new experiment path:
```bash
cp -r experiments/_template_multi_arm experiments/pcqm_<my_experiment_name>
```

### Step 2: Declare Prospective Trajectories
Under `experiments/pcqm_<my_experiment_name>/`:
- Create active prospective trajectories for each arm:
  - `arm_a/trajectory.json` (declares `trajectory_id`, `owner="desktop"`, `record_mode="prospective"`, `decision.outcome="ACTIVE"`)
  - `arm_b/trajectory.json`
- Declare `training_contract.json`.

### Step 3: Configure `terminal_bundle_spec.json`
Fill in `terminal_bundle_spec.json` using `terminal_bundle_spec.template.json` as a reference.
Specify:
- `arms`: Roster of candidate arms with unique identities.
- `candidate_artifact_paths`: Pointers to checkpoints, prediction manifests, etc.
- `trace`: Raw trace field mapping and cumulative axis semantics.
- `trace_manifest`: Comparability identity, exposure bounds, and reference target.

### Step 4: Pre-Launch Freeze
Before launching remote jobs, freeze all source identities into a prelaunch receipt:
```bash
python -m molgap.research_memory native-freeze \
  --spec experiments/pcqm_<my_experiment_name>/terminal_bundle_spec.json \
  --receipt experiments/pcqm_<my_experiment_name>/prelaunch_receipt.json
```
Commit the resulting files:
```bash
git add experiments/pcqm_<my_experiment_name>/
git commit -m "plan(exp): freeze prospective identities for pcqm_<my_experiment_name>"
```

### Step 5: Launch Guard Verification
Verify that the receipt is durably committed in `source_commit`:
```bash
python -m molgap.research_memory native-launch-guard \
  --receipt experiments/pcqm_<my_experiment_name>/prelaunch_receipt.json \
  --source-commit HEAD \
  --output experiments/pcqm_<my_experiment_name>/launch_binding.json
```

### Step 6: Execute Remote Multi-Arm Run
Run the training job on Kaggle / remote accelerator.
Download and verify durable artifacts (`best_model.pt`, raw `trace.json`, `completion_manifest.json`).

### Step 7: Record Raw Acceptance
Record no-inference acceptance for each arm in `raw_acceptance.json` (binding candidate artifacts, observed costs, and consumed roles).

### Step 8: Execute Generic Terminal Closure
Run the universal terminal closure entrypoint:
```bash
python -m molgap.research_memory terminal-closure \
  --receipt experiments/pcqm_<my_experiment_name>/prelaunch_receipt.json \
  --launch experiments/pcqm_<my_experiment_name>/launch_binding.json
```
*(Or simply run `python experiments/pcqm_<my_experiment_name>/close_terminal.py`)*

### Step 9: Inspect Replay Status
Check the machine-readable output in each arm's closure directory:
- `experiments/pcqm_<my_experiment_name>/arm_a/closure/closure_status.json`
- `experiments/pcqm_<my_experiment_name>/arm_b/closure/closure_status.json`

Verify whether each arm obtained `"replay_status": "REPLAY_READY"` or `"REPLAY_EXCLUDED"`.
