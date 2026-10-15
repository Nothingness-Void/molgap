# Common contract extraction acceptance

Snapshot: 2026-10-16. This records the formerly interrupted parser/IO extraction,
not the remaining full training-core convergence. The
[server handoff](SHARED_TRAINING_CORE_SERVER_HANDOFF.md) owns that specification;
the [Common API](../../src/molgap/common/README.md) owns the implemented surface.

## Scope

- Desktop candidate base: `d969e256`; server-compatible candidate base: `4aacdc58`.
  The latter is not a claim about the current remote server tree.
- Desktop implementation: `79f6a96a`, integrated by `c7ad10c8`.
  Server-compatible implementation: `0ea5cf18` on
  `codex/infra/common-contract-server`, for reviewed selective integration only.
- Extract strict Spec parsing/types into `common/spec.py`; keep reviewed family
  and addon declarations in `experiment_catalog.py`. Old imports and the exact
  wire-only `ExperimentSpec` snapshot remain compatible with existing guards.
- Extract canonical fingerprints, portable paths and immutable local IO into
  `common/wire.py`, `common/paths.py` and `common/io.py`. Legacy owners forward
  to these mechanisms. Add every required module to the static source inventory.
- Preserve default updates while adding the server's explicit clip-parameter
  argument and retained training-metric unit option to desktop. Add read-only
  single-plan validation; require explicit ownership and retain batch error indices.
- Seventeen staged mechanism/compatibility files have identical Git blobs across
  candidates, including all ten common Python modules and the new regression file.
  Catalogs, source inventories and historical terminal-owner policies may differ.

## Executed checks

Two independent `gpt-6-luna` / `max` workers used the project Python with
`PYTHONPATH=<candidate>/src` and CUDA/HIP/ROCR visibility set to `-1`.
Both verified `torch.cuda.is_available() == False`. No real-data model execution,
platform API, credentials, training, protected-role consumption or RML rebuild.

Initial grouped command: `python -m pytest -q` over these test files:

```text
test_common_contract.py test_common_api.py test_experiment_spec.py
test_experiment_package.py test_experiment_launch.py test_experiment_terminal.py
test_source_dependencies.py test_research_memory_plan_batch.py test_research_memory.py
test_family_training_core.py test_training_logging.py test_experiment_workflow.py
test_experiment_preflight.py test_experiment_prospective.py test_experiment_cli.py
test_kaggle_submission.py
```

| Snapshot | Passed | Failed | Skipped | Pytest seconds |
|---|---:|---:|---:|---:|
| Desktop, first pass | 941 | 94 | 8 | 379.18 |
| Server-compatible, first pass | 891 | 94 | 8 | 337.70 |

Both initial compileall and staged/unstaged diff checks passed. The first pass
was BLOCKED, not accepted. Failures were consolidated and fixed in one batch:

1. Extra Spec instance state broke strict launch snapshot guards. The facade now
   retains only `_canonical_json`; guards were not widened. Regression cases also
   reject injected catalog/override state.
2. The extracted publisher removed `launch.os`, the retained publication-failure
   test seam. That import is restored; publication still has one implementation.
3. The server's existing CLI whitelist omitted its existing `check-workflow`.
4. Desktop's isolated checkout lacked two ignored retained checkpoints. Main
   copies matched the existing mechanical receipts' SHA256 values:
   `faccd5f58a6ff5879cf41b39e3d9ba39778a83cfcb65b05d5c9e71f9b1e78eb0`
   and `6624ac73d5c4c1f25139a0bb7f4240f59b381a43cffa4982d39274eaa2d6ec79`.
   Only those exact bytes were copied into the isolated checkout; no model load.
5. Server preparation closure hashes failed after Windows CRLF conversion. LF
   restoration of fifteen files across preparation and two verification-recovery
   closures matched existing receipts exactly. Three scoped `.gitattributes`
   rules prevent recurrence. All five author-alignment receipts were inspected;
   no evidence/receipt Git content changed and no validator was bypassed.

Repair group: common-contract/API/spec/launch/CLI files, the previously failing
workflow case, and the previously failing RML cases. This is a focused rerun,
not a second complete run; overlapping counts must not be added.

The focused repair reruns passed the extraction/API/spec/launch/workflow/CLI
cases: desktop 352 passed / 1 skipped, server-compatible 358 passed / 1 skipped.
Their overall commands still failed four/three RML cases respectively while
the missing retained copy and later CRLF closures were being resolved.
Those metadata-only final checks are recorded separately below; no complete
green rerun of the initial sixteen-file command is claimed.

## Final disposition and publication

`CPU_WIRING_ACCEPTED` for the shared extraction. Desktop metadata-only follow-up
was 3 passed / 1 failed (55.01 seconds); the remaining failure was a stale exact
inventory assertion omitting the canonical ACTIVE single-EMA trajectory. The
test-only update explicitly asserts prospective/ACTIVE/empty evidence, without
changing its scientific status. Its final single-case rerun passed (13.59 seconds).

On real `molgap-desktop@c7ad10c8`, project Python ran
`-m molgap.research_memory check --frozen --portable`: exit 0, 31.596 seconds.
The validator reported frozen/current derived state and complete committed
evidence/runtime/derived/local-artifact closure under its existing policy.
No rebuild or blanket replay-ready claim is implied.

The server-compatible metadata follow-up remained 3 failed (29.65 seconds),
first stopping at the capacity-node G1 cost file. Do not represent its full
historical bank as accepted. A read-only audit of 85 finalization receipts found
923 published members: 387 match, 530 match only after LF conversion, six do
not match after uniform LF/CRLF conversion; none are absent. The six are
`prospective_snapshot.json` below these `experiments/<name>/rml_finalized/` roots:

```text
pcqm_k1_chem_typed_pair_token_100k
pcqm_k1_functional_group_token_100k
pcqm_k1_multiplicative_pair_value_100k
pcqm_k1_oneshot_triplet_pair_token_100k
pcqm_k1_sparse_triplet_100k
pcqm_k1_spd_pair_token_100k
```

Their committed Git-blob bytes also fail their existing receipt hashes. These
are retained-server-snapshot findings, not conclusions about current remote
results. Do not rewrite receipts, statuses or provenance to suppress them;
the server owner must reconcile authoritative source/history separately.
Beyond the fifteen exact-byte restorations above, this audit changed no records.

The desktop implementation is adopted; the server-compatible series is published
on `codex/infra/common-contract-server` for selective review. It includes the
earlier `c933990d` / `4aacdc58` caller work. It must not replace or wholesale merge
over the newer `molgap-server` branch. Current server private-resource absence
checks and research/runtime changes remain with their owner.

No native, scientific, cloud submission or new replay-ready qualification is
established by these CPU checks. Full accelerated training-core/addon/lifecycle
convergence remains the server implementation task, not an uncommitted local draft.
