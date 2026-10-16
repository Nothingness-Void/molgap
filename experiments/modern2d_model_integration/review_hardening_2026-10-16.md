# Targeted review hardening — 2026-10-16 JST

The supplied review of d4cc4bf524bf0dc5969a7ee2c27034107bcd0783 was
reproduced and addressed without scientific training. Scope was infrastructure
maintenance and generated CPU fixtures only. The [machine record](review_hardening_report.json)
owns test outcomes, failed attempts, source/archive pins and retained evidence.
Earlier [CPU integration](decision_cpu_2026-10-16.md), F1–F4 decisions, V2 data
acceptance and canonical RML were not revised.

## Reuse and exact changes

| Finding | Reused owner | Added behavior and scope |
|---|---|---|
| R1 observation/declaration gap | comparison_readiness, cross_scale_analysis, canonical RML metric semantics | Independently frozen per-run trace/run/target/metric/role/weight/checkpoint pins; report envelope checks |
| R2 metric-filtered LR | cross_scale_analysis | All-metric observed state history, same-step merging and conflict checks before selecting display metrics |
| R3 index-paired PMA residual | ESA attention/MLP modules and shared Session/Recipe/Kernel | Separate esa_small/2 CPU candidate; no input residual in query space; original esa_small/1 and author bytes retained |
| Indexed V2 preparation | accepted role manifests, PackedGraphDataset, role authorization and existing digests | Bounded synthetic-only read prototype; explicit offset maps, decoded identity checks and one retained chunk |
| PPGT ambiguity | unchanged PPGT constructor and author-pinned source inspection | Meta-device inventory and a dense-backbone lower-bound test; no module removal, width change or parameter sharing |

Training core, optimizer, scheduler, sampling, checkpoint state/recovery and
submit owners were not replaced. Optional family_version is serialized only
when explicitly supplied; old recipes retain their previous serialization
and default family versions. Its explicit override is limited to registered
CPU-synthetic interfaces. Same-shaped v1 tensors cannot resume under a v2
recipe because their frozen mathematical identities differ.

## R1/R2 acceptance interpretation

Both sides agreeing on an incorrect role, metric kind, target/unit, weight,
trace or checkpoint is insufficient. Each side is checked against its own
independently supplied frozen identity and canonical metric definition.
Raw-EMA spelling aliases are explicit; Clean-EMA is never treated as Raw-EMA.
SYSTEM_COMPARISON can select different live/EMA slots and learning rates only
when each obeys its own declaration and the symmetric tuning contract.
Auxiliary metrics also obey their own slot semantics; an EMA-selected run can
legitimately retain a live-development observation without relabelling it EMA.

All observed metrics contribute LR/exposure/run provenance. Duplicate metric
slots at one coordinate are rejected; different slots merge their state and
cannot hide conflicting LR/exposure/run identities. Missing observed state on
one side is not inferred from a declared schedule. A later conflict blocks the
later prefix while preserving an earlier fully matched prefix. No points are
interpolated. Missing historical pins remain CONTEXT_ONLY, never guessed from
agreeing rows. A selected checkpoint is bound at its selected step, not assumed
to be the final observed checkpoint.

The public comparator is a read-only report over trusted declarations and
observations. Its string/hash metadata is not a cryptographic attestation of
arbitrary caller-created rows. Accepted-trace import verifies retained bytes;
native output acceptance and RML admission remain separate owners. Neither
align_curves nor comparison_report can authorize training, publish scientific
acceptance or automatically reject 500K from 100K MAE.

## R3 version and limitations

The frozen author implementation's PMA conditional adds ordered edge-token
rows to learned-query rows only when their counts happen to match. The v1
counterexample was reproduced at 32 tokens under three finite noninitial
parameter states. These are normalized-output differences, not eV MAE or a
prediction of trained performance.

The separate v2 candidate applies attention from learned seed queries to
normalized edge tokens, followed by its query-space MLP residual. It never
adds token-index-paired input rows. Existing initialized attention, norm, MLP
and seeds are reused within each model, not shared across independent models.
Parameter count is unchanged. Version 1 and the raw author reference were not
silently patched or certified invariant.

Tests cover S−1/S/S+1, bidirectional S−2/S/S+2, multiple parameter RNGs,
initial/finite-stressed states, edge permutation plus atom relabelling, whole
output, PMA output, all parameter gradients and PMA input-gradient equivariance.
Exact uninterrupted/serialized-resumed Session states and read-only restored
evaluation are checked. Those properties are tested numerically within recorded
FP32 tolerances, not a proof of benchmark accuracy or GPU-kernel equivalence.

Edge-only input can omit isolated atoms in a graph containing bonds. Version 2
therefore explicitly rejects such synthetic inputs, as well as empty-edge
graphs, pending an authorized metadata census and a reviewed coverage policy.
No real member was dropped, substituted or reparsed. Supporting these inputs
through a new tokenization would need another explicit input/model version.

## PPGT author-code update, not a completed fidelity gate

[arXiv v4](https://arxiv.org/html/2504.12588v4) explicitly links the
[author repository](https://github.com/LiamMa/PPGT/tree/62cafd3abbb8d72e9a8b3146f947a192cf0aacaf),
pinned at 62cafd3abbb8d72e9a8b3146f947a192cf0aacaf (MIT). This is a new
source observation; it does not rewrite the earlier v3 search record.

Table 9 still lists small as 16×512, FFN ratio 2 and 17.6M. The released
attention constructor has separate full-width Q, KV and O projections and a
two-linear FFN. Under those widths the small backbone's weight-only lower bound
is 16 × (4×512² + 2×2×512²) = 33,554,432, before biases, norms, stems or heads.
The unchanged local constructor has 34,867,201 parameters. Thus ordinary dense
projections cannot satisfy all those table entries simultaneously; this does
not establish which author table/implementation entry is wrong.

Only a PCQM **base** YAML was present in the pinned config inventory: 12×768,
12 heads, FFN ratio 4, PE steps 8, two PE blocks, five SPE bases and mean pooling.
There was no released PCQM small YAML to reconcile the small count. Base YAML
also differs from Table 9 in batch/LR/weight decay/warmup. Author README warns
that the released norm requires PyTorch≥2.4 while some paper results used an
older environment. The source/config hashes and exact differences are retained
in the machine record; no released trainer, dependencies or data loader was
executed. Whole-family MODEL_FIDELITY_PENDING remains appropriate.

Further independent forward/gradient mapping must settle attention scaling,
the author's +1 multiplicative PE gate, residual layer scales, norm placement/
epsilon, RRWP/stem conventions, head and initialization. Author code availability
is not parity. No PPGT model change was made merely to match a parameter count.

## Indexed V2, safe ABI and private mirrors: preparation only

IndexedRoleDataset opens only generated CPU_SYNTHETIC_ONLY acceptance fixtures
for train_100k/train_500k/common_dev. Authority and role history are checked
before members/payloads. Byte pins, explicit ascending source-index maps,
non-contiguous offsets, decoded row/target/topology identities and physical
payload prefixes are checked. Audit/full/1M/official/test scopes, reparse paths,
unknown/3D schemas and real DATA_ROLE_ACCEPTED scope are denied. Hash or offset
tampering cannot silently pick another split. Only one packed chunk is retained;
fixture caps are 512 selected rows and 64 MiB per physical chunk.

This is **not** the released real-V2 family loader. It is not routed through
FamilyLoaders and does not relax contiguous V1 selection, the synthetic model
caps, CPU-only tensors, the existing feature-binding validator or any platform
gate. The accepted IMS master was not opened or rescanned. Existing V2 data
selection and molecular-identity acceptance remain unchanged.

The metadata-only mirror draft excludes independent_audit, private identity
SQLite and original source shards; only authorized train and Common-Dev role
bindings are candidates for export. Kaggle1/2 remain NOT_PUBLISHED. There is no
uploader, mount probing, credential use, dataset publication or job submission.

## Unexecuted release plan and remaining issues

1. Obtain explicit authority for real input qualification, its member scopes,
   metadata-only graph-size/edge/isolate census and measured preparation budget.
   Use already accepted indexed maps, not newly sampled development members.
2. Extend the owning FamilyLoaders input adapter for indexed source IDs without
   treating min/max as contiguous membership. Prove actual row/target/topology
   identity against accepted manifests and keep labels outside topology inputs.
   Resolve ESA isolate coverage without dropping molecules.
3. Freeze a real feature ABI binding base manifest, decoded rows/topology,
   role history, transform source/config and null/on-the-fly versus materialized
   feature identity. Test label/RWSE independence and forbid geometry/audit IO.
   Synthetic metadata proofs do not qualify actual features automatically.
4. Decide bounded on-the-fly or per-chunk derivation only after size/preprocessing
   evidence. Never allocate a full dense RRWP corpus. Freeze a separately reviewed
   GPU batching implementation and compare padding/mask/forward/backward to the
   CPU graph-wise oracle; do not just disable cpu_synthetic_only.
5. Obtain private export authority and exact account/dataset bindings. Inventory
   every physical export file for protected content, preserve or explicitly
   rebind decoded equivalents if repackaged, and retain native mirror receipts
   through the owning platform adapter. No mirror is considered published here.
6. After source/feature/fidelity/role release and a new compute decision, run a
   bounded optimizer-inclusive native BS128 qualification with declared device,
   precision/TF32/AMP, measured full allocation cost, memory, phase timings,
   interruption/resume and independently retrievable checkpoints. CPU durations
   are not native throughput, GPU costs or a hardware speedup estimate.
7. Resolve PPGT small source/config ambiguity and adapted ESA whole-family
   fidelity. Keep the separately documented legacy K1 package inventory defect
   outside these new model packages; do not delete a check for all-green status.

Outcome: INFRA_CPU_ACCEPTED for these bounded repairs and preparations only.
Real family fidelity/input/native release remains BLOCKED by the listed missing
evidence. Scientific status is NOT_EVALUATED; training remains
NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING. No STQS/early-stop, 100K veto, new seed,
500K/full experiment, model ranking or automatic successor was authorized.

## Primary-checkout delivery verification

The reviewed commits were fast-forwarded into local molgap-server. Primary
acceptance passed 293 tests with zero failures/skips and 14 unrelated CLI-help
cases deselected: 137 comparison checks, 107 model/indexed checks, 28 RML checks
and 21 navigation/layout checks. The separately retained shared-owner batch
passed 234; these batches are not summed into a whole-repository claim. Every
retained model archive was byte-verified again against the primary test result.
All 14 pre-existing dirty-file hashes, including the user's RML test, survived
integration and testing. Canonical RML and frozen model/reference files were
unchanged. Native preflight, scientific training and remote push were not run.

Primary navigation used its existing untracked historical documents and ignored
staging directory. That local success did not resolve the isolated checkout's
historical pointer dependencies or the separately documented legacy K1 package
issue. Their evidence and limitations were retained; all-green clean-checkout
or platform qualification was not claimed. Task execution stopped here.
