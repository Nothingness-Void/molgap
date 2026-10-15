# Shared training core: server implementation handoff

Snapshot: 2026-10-16. This is an engineering specification and source audit,
not a training release, completed-capability claim or replacement V5 contract.
Server implements; desktop pauses implementation and later reviews/integrates
the returned shared patch. No automatic monitoring or cross-machine handoff.

## Goal and decision

Keep two model backbones, K1 and GPTrans-T, with one reviewed addon interface
and one accelerated training lifecycle usable on both owner branches. Reuse the
four existing modules, not a third framework or a universal new model class:

```text
K1 / GPTrans backbone + registered addon + typed recipe + accepted data identity
  -> shared update / state / evaluation / resume / logging / profiler lifecycle
  -> immutable platform-neutral package and existing evidence protocol
  -> Kaggle skill + adapter OR IMS skill + adapter
```

Converge the active paths in one bounded project with staged acceptance. Do not
rewrite the repository or preserve multiple live trainers indefinitely. Existing
accelerated execution stays the base; historical slow code is frozen replay
material, not a new selectable default. Names and layout below are navigation,
not requirements to rename working APIs or rebuild them to match this document.

## Source baseline and reuse

Refresh refs and inspect actual branch/HEAD/status before starting. The inspected
snapshots are desktop `d969e256ba217178cfa30991e4de0cce38d94ec5` and remote server
`21a76c1153236d1d811d5ebba0839f83b2471226`; later commits require delta review.
The [dated infrastructure audit](SHARED_INFRASTRUCTURE_AUDIT.md) is an earlier
snapshot, not the current implementation inventory.

The already-written [Common API](../../src/molgap/common/README.md) is reusable:

| Commit / owner | Reuse boundary |
|---|---|
| Desktop `0edc7bdbea2cee3759758b50c9b7981b219543df` | Seven-file host-neutral API extraction: declarations, strict identity, explicit-account SDK compatibility, pinned atomic downloads, aliases to existing runtime owners. Reuse this patch rather than implementing it again. |
| Desktop `d969e256` | Desktop callers, source inventory, docs and `tests/test_common_api.py`; review caller-specific changes, do not wholesale apply desktop authority to server. |
| Local server `c933990d` / `4aacdc58` | Same common extraction and server caller conversion, CPU-tested before subsequent remote changes. Local-only at audit time; not evidence these commits exist on origin. Do not reset remote server to these tips. |

The desktop publication accompanying this handoff supplies the common extraction
and shared tests through Git. Verify advertised refs are fetchable before using
them. Do not merge the entire desktop branch into server for these helpers.
New remote private-resource absence handling must survive SDK consolidation:
an authenticated bounded owner listing is required before treating a private
403 as absence; ambiguity must not authorize duplicate publication.

At the inspected snapshots these five files are byte-identical across desktop
and remote server: `model_substrate.py`, `family_training_state.py`,
`fast_sandbox.py`, `source_dependencies.py`, `shared_model_adapter.py`.
Keep them and their existing registries. `common/runtime.py` aliases these owners;
it does not implement a trainer. Earlier common adapter checks passed on CPU;
that is not acceptance of the remaining convergence or native platforms.

Uncommitted desktop worktrees `common-contract` and `common-contract-server`
contain an interrupted, untested extraction attempt. They are not accepted
dependencies or the implementation baseline. Do not copy them as completed code.

## Gaps to close

| Finding | Required treatment |
|---|---|
| Generic ordered addons use the `graph_screen_training` factory ABI; accelerated family dispatch still uses fixed modes. | Connect reviewed composition to the existing accelerated lifecycle; prove actual optimized forwards reach the slots. Unsupported compositions fail closed, not silently become a baseline. |
| `FamilyKernel` imports model construction from `pcqm_gptrans_v4` and K1 forward/update helpers from `k1_screen_training`. | Extract reusable leaf model/input/forward/loss helpers first. Then thin legacy entrypoints. Reversing this order can create circular dependencies. |
| All eight core/data/recipe/kernel/loaders/session/runner/outputs files differ across branches. | Review semantic differences, not just file hashes. Reconcile evaluator ABI, explicit clip parameters, units, data confinement, initialization identity and completion handling. |
| Server modes add development reads, segmentation, caps and alternate objective/source bindings. | Keep question/role/selection policy explicit. Do not silently change a frozen mode's meaning or infer policy from hostname, branch or dataset size. |
| Remote `gptrans_execution_diagnostic` adds a manual training loop; generic Kaggle runtime directly imports its phase/device/artifact logic. | Reuse bounded iteration and observation hooks in the existing lifecycle. Expose reviewed neutral phase descriptors at the platform boundary; keep GPTrans attribution/finalization experiment-owned. |
| Clean state is optional and not fully integrated into session selection/checkpointing. | Integrate explicit state actions and endpoint policy without enabling calibration, EMA or extra evaluation implicitly. |

The old K1 and GPTrans trainers already reuse shared update/outer-loop primitives;
do not describe them as entirely independent implementations. Remaining duplication
is largely lifecycle, evaluation, selection, resume and orchestration ownership.
Keep special observer needs explicit: pre/post clipping, bounded profiling,
synchronized timing and state capture must not rely on global monkeypatches.

## Implementation boundaries

- Reuse `ExperimentSpec`, `TrainingAdapter`, `TrainingAddon`, artifact adapters,
  `ModelSubstrate`, `FamilyKernel`, `FamilyTrainingSession` and the existing V5/RML
  owners. Extend the nearest owner only for a demonstrated missing capability.
- Use structured, validated configuration. Registered code selects factories and
  hooks; JSON must not become arbitrary import/callback execution. Attach addons
  in declared order and seal before optimizer construction; retain stable weight
  keys and check every owned trainable parameter reaches the optimizer.
- Dataset size, budget, seed, precision, loss, optimizer, EMA, calibration,
  evaluation cadence, selection, profiler budget and resume policy are explicit
  contract/config facts, not launcher defaults that override the submitted plan.
  This does not authorize changing existing contracts to improve utilization.
- Preserve RNG, batch order, target transform, optimizer/clipping order, EMA update
  count, tail handling, dropout/BN behavior, metric units and resource accounting.
  Optimizations with changed numerics need explicit evidence, not an equivalence label.
- Keep one evaluation/state implementation for supported new family sessions.
  Existing public entrypoints may forward to it; frozen historical loaders and
  payloads remain usable under their original source identity.
- `molgap-family-training-state-v1` is the existing weight-state component,
  not the full resumable checkpoint. Reuse/extend the session envelope to retain
  optimizer, scheduler, RNG, cursor, exposure, partial metrics, best selection,
  BN buffers and source/config/data identities. Raw/Clean absence is explicit;
  Clean becomes stale after its source changes. Update EMA exactly once per step.
- Platform adapters may own authentication, mounts, environment setup, scheduling,
  timeouts, upload/download and receipts. They must not own a second training loop.
  Keep standard logging and profiler owners instead of adding experiment printers.
- Preserve running/frozen jobs, scientific decisions, canonical evidence, source
  archives and production defaults. Do not rewrite history or manufacture RML
  readiness to make a migration look complete.

See the [four-module interfaces](MODULAR_EXPERIMENT_RUNTIME.md),
[family core](FAMILY_TRAINING_CORE.md), [workflow](EXPERIMENT_WORKFLOW.md),
[logging](TRAINING_LOGGING.md) and [reuse map](EXPERIMENT_ADDON_GUIDE.md).
These documents own their APIs; update them rather than duplicating their manuals.

## Platform scope

Reuse `prepare_family_training` for neutral staging and the existing Kaggle
adapter/submission facility. Reuse `platforms/ims/v4_submit` and its workload
skill for IMS; do not create a replacement scheduler or credential layer.
At audit time the legacy workflow platform registry contains only Kaggle.
Neutral staging is not proof that IMS is wired or natively qualified.

The shared recipe defaults to Torch 2.11.0, while the checked-in IMS registry
declares Torch 2.7.1+cu126. This is repository metadata, not a live IMS query.
Select a supported explicit runtime and report unresolved operator/ABI requirements;
do not disable exact runtime validation. Read `platforms/REMOTE_HANDOFF.md` before
any IMS command. This handoff grants no remote submission, GPU/DCU work, protected
role access, full training, new scientific experiment or V5 redesign authority.

## Acceptance and return package

Batch coherent implementation first, then run one grouped CPU cross-suite;
repair and rerun affected failures instead of rerunning everything after every
small edit. Use synthetic fixtures only. Reuse/extend the existing tests:

- `test_model_substrate`, `test_family_training_state`, `test_fast_sandbox` and
  `test_source_dependencies`: ordered addons on both accelerated families, real
  slot reachability, gradients/parameter ownership and packaged dependency closure.
- `test_family_training_*`, `test_family_full_lifecycle`: both family recipes and
  synthetic 100K/500K/full policies share the lifecycle; missing roles stay unread;
  Live/Raw/Clean evaluation and state survive failure, calibration and roundtrip.
- Extend the existing mid-pass resume test: uninterrupted versus resumed updates
  preserve model/BN/EMA, optimizer, scheduler, RNG, order/exposure and metrics;
  changed/corrupt identities reject without partial mutation. Test cross-owner
  compatibility using the same package/runtime, not two artificial host parsers.
- `test_common_api`, affected Spec/worker/package/workflow/prospective/terminal,
  `test_kaggle_submission` and `test_ims_v4_submit`: preserve fail-closed identity,
  native-proof, role and cost semantics, no credentials/network in CPU fixtures.
  Fake platform APIs must enforce actual SDK signatures. Run isolated package
  imports, not only AST discovery. Include compile and diff checks.

Cross-platform checkpoint readability is required; bit-exact continuation is
assertable only in a controlled same-runtime deterministic test, not across
different GPU models, Torch versions or numerical backends. CPU wiring does not
prove T4/A100 speed, precision retention, VRAM or cloud submission. Native checks
remain separately authorized and explicitly pending when not executed.

Return one concise integration report with exact branch/HEAD and clean/dirty
status; shared commits separated from research/host-specific commits; reusable
API and caller map; capability matrix and unresolved gaps; exact test commands,
pass/fail/skip counts and native checks not run; checkpoint compatibility/migration
notes; provenance and untouched frozen-job boundaries. Supply one minimal example
per family showing config + small addon -> sandbox -> package -> platform handoff,
without submitting anything or claiming scientific/RML acceptance from fixtures.

Common mechanism files must be identical in both owner branches, enforced by a
shared compatibility suite and a reviewed patch, not a promise about folder names.
Registration/policy may differ explicitly. Provide the same shared patch to both
owners; desktop reviews and selectively integrates after server completion.
Do not wholesale merge server research or import either side's controller into
the core. Reduce docs to API owners and short pointers; no AGENTS/V5 expansion.
