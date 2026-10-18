# Modern2D modularity review — 2026-10-18

Scope: read-only implementation review of PPGT and ESA against the human's
modularity, ordinary-addon and minimal-workflow goals. Source HEAD was
`6265fb8e`; this report does not change code, frozen experiments or release
authority. No remote work, real-data loading or model training was performed.

## Assessment

The shared training foundation is reusable. The Modern2D extension surface is
not yet a simple addon workflow. It is an explicitly bounded reference-screen
integration with experiment-specific constraints embedded in shared modules.
Keep the core and reduce the coupling around it; another trainer, plugin
framework, evidence system or platform submitter is unnecessary.

The reform plan's original ordinary-addon acceptance examples covered K1 and
GPTrans. Its initial Modern2D scope was CPU-only. The gaps below concern the
subsequent native/formal integration and the desired future extension surface;
they do not retroactively invalidate the earlier bounded CPU delivery.

## Findings

### 1. Ordinary addons are blocked at several independent layers

- `experiment_input.py:222` accepts only K1/3 and GPTrans/2 in the scientific
  configuration compiler.
- `models/substrate.py:28,125` restricts slot bindings and substrates to those
  two families.
- `experiment_execution.py:152` rejects addons in Modern2D preview/formal
  adapters; `family_training_recipe.py:136` requires reference mode and no
  addons; `modern2d_training.py:38` repeats the restriction.
- `family_training_kernel.py:55` selects addon interfaces using a K1-versus-
  other expression, rather than `recipe.interface`. Simply removing the guards
  would not make later ESA versions route correctly.

Consequence: implementing a small readout module and adding one catalog entry
is insufficient. Several shared owners must be edited before a normal candidate
can be declared. This is the highest-priority extension-design gap.

Recommended bounded change: extend the existing substrate with reviewed
PPGT/ESA readout bindings first, preserve the existing addon parameter/state
ownership mechanism, and derive supported compiler/recipe behavior from the
single catalog. Add deeper message/attention slots only when a concrete module
needs them; the two model representations need not share an artificial payload.

### 2. One experiment's policy is compiled into family execution

`modern2d_training.py:16-59` fixes 100K, 60 passes, seed42, reference-only,
no EMA and Common-Dev live selection. Its context additionally fixes Kaggle,
two T4 devices and the exact two-family combination (`:83-93,145-169`).
`modern2d_training_release.py:159-166` fixes LR=.001, weight decay=1e-5,
warmup=4 and the schedule. `platforms/kaggle/run_experiment.py:90-93,122-142`
selects the runtime/deadline through the exact PPGT/ESA pair.

These checks correctly bound the authorized historical screen. As the sole
supported family path, however, they force infrastructure edits for a different
approved horizon, learning rate, single-family run or pairing. A new model
version is also used to represent a new execution admission although the model
implementation itself is unchanged.

Recommended change: retain legacy readers and frozen checks. For future runs,
let the existing recipe and run authority supply horizon, roles, resource cap
and composition, and validate their agreement generically. Keep model version,
runtime capability and per-run authorization distinct. Configuration support
must not itself authorize a new run or loosen an old experiment's bounds.

### 3. Preparation and terminal translation contain campaign instances

`modern2d_training_prepare.py:23-37` contains the exact account, kernel, dataset,
previous retry directory and E-drive paths. Its preparation reads the prior
qualification campaign directly (`:75`). `modern2d_training_records.py:25-49`
pins one failed run's source, package and input hashes.

These are useful reproducible historical adapters, but they are not general
family interfaces. Future candidates should not need copies of them.
Move new run-specific values into ordinary experiment inputs/machine profiles;
reuse existing prepare/package/journal/inspection/RML transactions. Keep exact
old failure adapters callable for their original records rather than deleting
their provenance or treating them as the default route.

### 4. Integrity checking remains inside the random-access hot path

`data/indexed_inputs.py:185-204` retains only the current shard. Switching
shards invokes `_pin`; `pcqm_unified_roles.py:256` hashes the whole file.
The retained [loader probe](../modern2d_100k_formal_screen/retry_v2/results/loader_risk_2026-10-18.md)
demonstrated this using synthetic data. Its serial one-cache model projected
50,181 pin/hash/open calls and approximately7.79TB of hash input for one
seed42 first pass. Those are modeled bytes, not measured physical IO or remote
elapsed time.

Recommended change: separate verified immutable shard identity from residency
of mmap handles. Verify each shard once per appropriate process/input lifetime,
retain bounded handles, and invalidate/reverify when file identity changes.
Keep the sampler, source members and label identity unchanged. The exact
invalidation/immutability contract needs targeted mutation and spawn tests;
blindly removing hashes or sorting the sampler would not address the design.

### 5. Feature identity is broader than the feature transform

`modern2d_feature_binding.py:27-43` includes the model source module in
`transform_code_sha`; `build_indexed_feature_binding` hashes the entire model
configuration into `transform_config_sha`. A readout-only edit can therefore
change the feature binding despite unchanged topology/RRWP/masks.

Recommended change: separate transform-relevant source/configuration from
model source/configuration while retaining both in the full experiment
identity. This permits reuse of unchanged accepted inputs without weakening
model provenance. The observed coupling does not prove graphs are rebuilt;
the current path derives features on the fly.

## What already follows the design

- `ppgt_author.py` and `esa_native.py` contain model implementations and typed
  configuration; their `make_model` factories are independent of submission.
- `family_training_kernel.py` uses those factories and the shared optimizer
  and normalized-Gap update. `training/session.py:383` invokes the same
  `run_training_loop`; sampler, checkpoint and restore owners are reused.
- Model features have separate modules, and data is consumed through a shared
  accepted indexed view. Output inspection, packaging and submission use
  existing owners rather than new full frameworks.
- Source/data/role identity, immutable receipts, checkpoint consistency and
  full-allocation cost remain useful checks. Simplification should reduce
  duplicated policy and manual declarations, not discard these facts.

## Separate performance qualification still needed

PPGT's `forward` (`ppgt_author.py:294`) iterates graphs individually and calls
RRWP derivation inside each `graph_forward` (`:271`). ESA (`esa_native.py:160`)
already pads tokens and runs batched attention. This is a concrete difference
in execution design, not proof of a measured speed ratio. Profile the existing
feature, forward/backward and loader stages after correcting the loader risk;
consider bounded batching or transient feature reuse without creating a large
unrequested dense dataset cache. Preserve author-equivalence tests.

## Smallest useful completion criterion

For each family, one existing-style readout addon should require only its
module, one catalog declaration, focused tests and experiment configuration.
It must pass CPU forward/gradient/parameter ownership, save/restore and the
ordinary configuration-to-prepared-package path. Changing a supported LR or
horizon should alter configuration, not platform or training source. A warm
cross-HEAD continuation must reuse the original frozen identity. These are
local infrastructure checks and do not require another training allocation.

Priority: remove addon/experiment-policy coupling and fix the loader hot path;
then narrow feature identity and qualify steady-state model execution. No
wholesale rewrite is justified by this review.

## Delegated CPU verification

The configured Luna worker reported37 passing existing catalog, registration,
Modern2D-registration and substrate checks in19.87 seconds. These confirm the
present registered boundaries; they do not establish Modern2D addon support.
The additional scientific-input control was bounded and marked INCOMPLETE for
this review. Its captured output included two passes, but no additional
acceptance count is claimed. The worker confirmed its pytest session had exited
and no further suite was started. No repository source or test file was changed.
