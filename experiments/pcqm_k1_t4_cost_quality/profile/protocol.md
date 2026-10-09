# Bounded native T4 execution diagnostic

The parent owns the prospective contract, release and platform reconciliation.
Preparation requires its pinned ACTIVE prospective record; it neither releases
the later quality experiment nor changes a model recommendation.

Reuse the accepted A100 profiler case loop, native500K selected epoch49 loader,
family forward, FP32/noTF32 seed42 settings, atomic IO, source allowlist/path
checks and immutable freeze helper. Prefer exact-byte reuse of the accepted
A100 `train_probe.pt`; no graph decoding or pickle loading during preparation.
The frozen source inventory must include `pcqm_wedge.py` (serialized WedgeData).
No model/trainer/loop copy, accepted-checkpoint rewrite or reference retraining.

Two sequential scratch cases on cuda0, both workers2, physical batch128:
single normalized L1 and double_mean2 (mean of two supervised L1 losses,
consistency coefficient0). Each independently reloads selected state, resets RNG
and creates AdamW lr4e-4/weight_decay1e-5/foreachFalse/fusedFalse, clip1.
Five synchronized warmup steps and24 synchronized measured steps per case.
FP32 without TF32; no geometry, scheduler, development evaluation, trained-model selection or
quality proof. Single and double passes differ in stochastic/BN behavior.
The frozen old objective has no `consistency_weight` argument: this adapter
reuses its returned `supervised_l1` component, never its combined default0.1
loss for backward. The unused disagreement computation remains in forward timing.
Scratch optimizer steps are real (29 per case); NO_TRAIN means no scientific
training/selection, not an assertion that no optimizer ran.

An independently reset mean2 scratch state runs the shared synchronized phase
instrumentation: eight additional real optimizer steps, first two warmup and
six recorded observations. Separate loader/H2D/forward+loss/backward/clip/
optimizer timings are instrumentation, not throughput. Total scratch updates66
(29+29+8), with no selected-state gradient-relation probe; gradient-norm reporting
is optional and unmeasured. The A100 default uses this same extracted helper,
with its existing objective and phase artifact format unchanged.

Four additional eval-only batches (512 training members) use a fresh selected
scratch state, no gradients or optimizer steps. Record synchronized loader/H2D/
forward wall time, not accuracy. One atomic scratch checkpoint write records
local-filesystem serialization/publication seconds and bytes. It does not write
the accepted checkpoint. Actual full50K development and remote-upload times
remain unknown; no sample-to-full extrapolation.

Exactly the accepted fixed4096 unique pure2D rows in drawn order from500K
official-train-derived members; no development or protected roles. CPU prepare
pins the exact retained sample bytes and accepted manifest row order. No redraw,
new shard decode or silent replacement. Worker verifies the
caller-pinned payload manifest and all source/payload bytes before Torch loads.
Parent must freeze and review the complete transitive source inventory; runtime
package verification is not acceptance of arbitrary supplied pickle content.

Worker supervisor timeout900seconds includes child imports, verification, load,
execution and teardown. Package `run.sh` timeout1200seconds includes optional
setup. Requires observed two native T4s; worker allocation is elapsed times
actual visible count. Active-case time uses one executing device's case wall
time including loader, not measured GPU busy. `active_work_T4_device_hours`
also includes the phase and train-eval/localFS-write windows on cuda0, each
including independent scratch load and teardown; these are not GPU-busy hours.
Setup allocation remains missing
until parent observes it; overall ceiling1200*2/3600=0.666667 T4-device-hours.
Never extrapolate global training cost or interpret timing as accuracy evidence.
Timeout/partial artifacts are incomplete, never a successful diagnostic.

## Parent prepare plan

Ordinary JSON: `trajectory` `{path, sha256}`;
`plan_receipt` the pinned RML PLANNED receipt matching the trajectory ID;
`accepted_profile_manifest` the pinned accepted A100 manifest (same path/SHA
binding shape); its checkpoint, sample order and frozen owner sources must match;
`checkpoint` `{path, sha256, source_sha256}`; `train_probe` `{path, sha256}`;
`source_root` the retained accepted payload; `source_files` mapping POSIX
`src/molgap/*.py` paths to exact byte SHA256; `source_overrides` pins only the
reviewed profiler and lightweight `evidence_pointers.py` additions;
`packaging_files` pins bootstrap, shell, setup, entry template and this protocol; `source_input_commit`
is HEAD at generation, while file hashes also bind uncommitted reviewed additions.
`sample_source_idx` the accepted4096 indices;
`role="official-train-derived-500k"`, `geometry_used=false`.
No default local paths, implicit sample redraw, credentials or remote commands.
Parent publishes the trajectory through the existing RML `plan` API before
preparation, not by writing an unvalidated JSON substitute. Preparation uses
current checkout metadata/hash/immutable-copy helpers only; it does not import
or execute a frozen family reader/model. Runtime uses only payload `src` and
rejects any imported MolGap source outside the pinned inventory. All old A100
source files except the profiler are retained unchanged, including the old
consistency objective, native500K loader and serialized WedgeData dependency.
Runtime payload verification reuses the lightweight root-confined
`evidence_pointers.resolve_repo_pointer`, avoiding package/spec/launch/RML imports.

Metadata-only default plan generation, after parent RML planning/review:

```powershell
.\.venv\Scripts\python.exe experiments/pcqm_k1_t4_cost_quality/profile/prepare.py --accepted-payload D:/w/k1-colab-profile/platforms/_records/colab/staging/k1-profile-a100-20261007/payload --trajectory <planned-trajectory.json> --plan-receipt <plan-receipt.json> --output <new-plan.json>
```

```powershell
.\.venv\Scripts\python.exe experiments/pcqm_k1_t4_cost_quality/profile/prepare.py --plan <parent-plan.json> --output <fresh-payload>
```

After parent verification/publication, the Linux package command is:

```bash
python /payload/bootstrap.py --root /payload --output /durable-output --expected-manifest-sha256 <payload-manifest-sha256>
```

Parent retains exact setup/process/allocation observations and independently
retrievable runtime, per-case, result, completion or failure artifacts. This
adapter neither publishes nor certifies durable platform retention.
Optional setup requires both `--setup /payload/setup.sh` and its
`--setup-sha256 <digest>` at bootstrap. Overall1200seconds includes verification
and setup; bootstrap gives the process-group shell timeout the remaining budget.
Worker has its independent900second supervisor. Package process provenance is
printed even on worker/setup failure, for the parent's existing adapter log
retention; worker process observations are atomic artifacts. The retained A100
sample's existence is not a new scientific or platform runtime qualification.

## Kaggle entry

Parent preparation copies the template `kaggle_entry.py` to the kernel entry
and replaces the manifest, setup, archive and extractor digest markers.
The private dataset contains a flat `source_payload.bin`, outer manifest and
pinned stdlib `unpack.py`. The existing safe source extractor verifies the
inner manifest and exact file inventory; uploaded directory layout is not assumed.
Retain the resulting entry SHA separately in parent metadata; never patch files
inside the already manifest-bound payload (which would create a circular pin).
No API or publication is performed by these scripts.

Entry discovers exactly one manifest with native-T4 format under `/kaggle/input`,
verifies its digest and setup/bootstrap/run.sh bytes before executing any of
them, then observes exactly two T4 names using nvidia-smi. The worker separately
checks Torch-visible device count. The allocation clock starts before discovery,
verification or installation, and the outer process-group timeout plus the
bootstrap's remaining-budget argument include all installation/execution work
within1200seconds. Setup uses the standard bootstrap dependencies only:
`numpy<2`, `torch-geometric==2.6.1`, `ogb==1.3.6`; installed Torch is unchanged.

Entry atomically retains and prints `entry_observation.json`, including full
entry wall time, observed GPU names/count, allocated T4-device-hours, bootstrap
wall time including setup, immutable input digests and complete/incomplete state.
GPU-busy remains unknown. Failure before GPU observation keeps allocation count
and device-hours unknown, not zero. Parent owns durable Kaggle output retrieval.
Local entry execution after staging pins is `python <pinned-kernel-entry.py>`;
no such real execution occurred during implementation.
