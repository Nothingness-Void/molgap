# Modern2D versioned input and Session integration

This guide describes reusable integration APIs, not a scientific training or
native release. The [dated evidence](../../experiments/modern2d_model_integration/integration_attempt_2026-10-17.md)
owns verification and missing qualifications. Historical CPU versions and data
identities remain readable and unchanged.

## Owners and versions

| Interface | Model/input change | Execution boundary |
|---|---|---|
| `ppgt_small/1`, `esa_small/1`, `esa_small/2` | Frozen synthetic prototypes/reference and PMA correction | Original CPU-only restrictions |
| `ppgt_small/2` | Pinned author-code PPGT core, explicitly adapted budget configuration; author RRWP/SPE ABI v2 | Shared Session integration preview |
| `esa_small/3` | ESA/2 invariant PMA plus directed-bond and isolated-atom tokens, padding-aware masks | Shared Session integration preview |

Select the new version explicitly with `FamilyRecipe.family_version`; defaults
do not upgrade old callers. Model configuration is typed and enters recipe and
checkpoint identity. Inputs use `modern2d_inputs.validated_topology_views`;
labels and retained RWSE are not consumed by either model. No geometry is
accepted. Oversized graphs reject the entire request, never drop members.

The new families reuse `FamilyKernel`, `FamilyLoaders`, `FamilyTrainingSession`,
the optimizer/schedule/sampler/state owners, Logger, Profiler and SessionOutputs.
They do not bind GPTrans execution layout. Preview CPU output inspection reports
`INTEGRATION_PENDING`, not runtime qualification or scientific acceptance.
Workflow/native dispatch and Kernel CUDA remain closed pending actual release.
This delivery does not yet supply an admitted native-preflight route for these
versions. That next adapter step must reuse the existing qualification/session
owners after exact real-input, source, authority and resource bindings exist;
it must not turn off the preview guards for a formal training push.
The new preview recipe also pins its feature-transform source closure; changing
the RRWP/mask/helper implementation changes recipe identity and invalidates an
old serialized policy. Indexed views expose a source/config/row/topology/history
feature binding, checked against an independently frozen expected binding.

## Accepted indexed inputs

`AcceptedIndexedInputPermit` is separate from the frozen synthetic permit.
It accepts only `train_100k`, `train_500k` and `common_dev`, with externally
frozen acceptance, role-manifest, version, authority and role-history pins plus
an explicit chunk-byte ceiling (at most 512 MiB). Wrong scope/count, protected
roles and mismatching authority are refused before graph payload access.

The owning data record supplies these pins. Reading a mount and copying its
self-declared hashes into a permit is not independent acceptance or authority.
The loader trusts the pinned independent parent acceptance and verifies payload
bytes and decoded selected IDs/labels; it does not redo molecular selection or
recollate all real graphs to replace the parent's topology acceptance.

The existing `load_training_data` detects the small frozen input binding:

```python
binding = {
    "format": "molgap-indexed-training-input-v1",
    "data_root": "indexed_assets",
    "scope": "ACCEPTED_INDEXED_INPUT_V1",
    "train": {"permit": asdict(train_permit), "authority": train_authority,
              "role_history": train_history},
    "development": {"permit": asdict(dev_permit), "authority": dev_authority,
                    "role_history": dev_history},
}
```

Freeze its actual bytes as `DatasetSelection.manifest_sha256`. `data_root` is
relative to that binding, with no traversal/junction fallback. TrainingData
retains exact `train_source_indices` and `development_source_indices`, sets
`indexed=True`, and uses `None` for both contiguous source starts. The shared
evaluator checks these exact ascending IDs, not `arange(min_id, max_id)`.
Normalization is computed from selected train labels only. A train-only read
does not open the Common-Dev payload. There is no old-development substitution.

One mapped chunk is retained per dataset/worker. Returned graphs are cloned;
PyG's per-graph memoization is cleared, and spawning does not pickle a loaded
chunk. Full-chunk byte hashing remains CPU IO; none of these checks establishes
native loader throughput.

## Safe private export

`pcqm_indexed_export.export_indexed_roles` accepts explicit parent pins, typed
permits, authorities/history, chunk budget and per-role row ceilings. Select
exactly one training scale plus Common-Dev. It filters/collates existing tensors
without graph rebuilding, reparsing, new splits or dense corpus feature caches.

Each exported chunk must match its parent's decoded row, target and topology
hashes. Chunk grouping and logical identity stay fixed; new physical byte and
manifest identities are explicit. Clone selected tensors before serialization
to avoid retaining unselected backing storage. Atomic receipt-first chunks are
resumable; a completed export is rechecked, never repaired or overwritten.
`verify_indexed_export` independently checks a separately pinned export receipt.
Any mismatching content retains failure evidence and cannot publish acceptance.

The exact output allowlist excludes Audit/1M/Full roles, the private identity DB,
original source shards and stray files/directories. Training payload folder names
retain the loader ABI but contain only requested training members. A label-free
census reports graph sizes, edge/isolate coverage and cap violations without
accessing or reporting target values. Scratch stays beside the explicit output,
not in system temporary storage. Platform access must additionally obey its
authorized root and scheduled high-memory CPU rules.

Export acceptance preserves synthetic versus accepted-parent scope. Neither an
export receipt nor successful CPU fixtures publish a Kaggle mirror. Publication,
remote byte verification and account binding belong to the existing platform
adapter/workload skill; no uploader or submitter was added here.
The thin `experiments/modern2d_model_integration/scripts/export_inputs.py` entry
accepts one externally frozen request and an explicit filesystem boundary. It
does not touch payloads by default; `--execute` performs only the CPU export.
On IMS the boundary must be the authorized user root from `REMOTE_HANDOFF.md`,
and execution must run in a scheduled high-memory CPU job, not on the login host.

## Bounded qualification adapter

`modern2d_input_preparation.prepare_qualification_inputs` freezes portable
indexed bindings, selected-train-only FP64 normalization, membership tensor
digests and default-version feature metadata from independently pinned selected
exports. Run real payload work on scheduled CPU; outputs remain separate from
the immutable export. It constructs no model and derives no tensor features.

`modern2d_qualification.QualificationProbe` is a distinct train-only recipe:
seed42, BS128, exactly two optimizer steps on each of three repeat/resume paths.
`Modern2DQualificationContext` binds source/package/account, authority, accepted
100K/Common-Dev permits and train-feature identity. A process-local expiring
admission permits at most six scratch updates. The shared Session, sampler,
optimizer and serialized state remain their existing owners. No Dev evaluation
or formal continuation is available through this admission.

The thin Kaggle bootstrap and `modern2d_qualification_runtime` supervise only
those two independently isolated workers, the pinned environment and an atomic
whole-allocation ledger. Environment startup and idle devices count inside the
same thirty-minute allocation; unknown pre-Python provisioning stays unknown.
Qualification certificates are distinct from screen certificates and are
explicitly rejected by formal training. Synthetic results are separately marked
`CPU_SYNTHETIC_VERIFIED`, never GPU-qualified. Actual releases and observations
are recorded in the owning [bounded task](../../experiments/modern2d_model_integration/native_preflight/README.md).

`kaggle_mounts.resolve_dataset_mount(input_root, dataset_source, *,
mount_name=None, marker=None, report=None)` resolves an explicitly bound
`owner/slug` at `input_root/slug`, `input_root/owner/slug` or
`input_root/datasets/owner/slug`. These are three supported transport locations,
not evidence of the failed worker's actual layout. It never enumerates
directories or reads payloads. Multiple roots, missing markers, namespace/slug
drift, symlinks and junctions fail closed, with bounded discovery diagnostics.
Flat paths do not encode an owner: ownership is not inferred from their names;
the frozen source identity and content/authority gates remain necessary.
The owner-qualified fallback is documented for underlying benchmark notebook
attachments in the [official CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/benchmarks.md);
that source does not establish the failed V1 worker's actual layout.
The preparation path and parent/worker data validation share this resolver;
launch/source/acceptance/role identities remain independently hash-checked.

Before the source is available, the prepared entry embeds the exact packaged
stdlib resolver and pins its source-dataset identity. It verifies launch bytes
before parsing, then the archive digest and embedded resolver's archive bytes
before executing the runtime. Failures retain their phase and discovery
observations. Bootstrap timing is explicitly `PYTHON_OBSERVED_WINDOW_ONLY`;
unknown provisioning, queue and complete allocation/billing cost stay null.
A CPU fixture or repaired entry grants no remote retry or training authority.

## Remaining native/scientific boundary

Each release needs a scheduled, budgeted export/census and private mirror
verification; actual status belongs to `CURRENT_STATE.md`. Feature timing and BS128 GPU numerical/optimizer/resume,
memory/throughput and full-allocation cost are unmeasured. Then freeze the
two-arm comparison, initial states, tuning/selection and resource contract;
use the existing platform journal. No LR/horizon is silently inherited from
another family, no baseline is retrained automatically, and no 100K score or
uncalibrated early-stop rule may automatically veto 500K.
