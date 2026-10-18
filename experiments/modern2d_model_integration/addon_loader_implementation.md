# Modern2D addon and indexed-loader implementation

## Scope

The local implementation reuses the existing catalog, model substrate,
FamilyRecipe, FamilyKernel, FamilyLoaders and Session. It adds no model body,
training loop, optimizer, sampler, recovery engine or submitter.

PPGT execution interface 3 and ESA execution interface 4 expose only the
`pre_readout` slot, through `ppgt_readout_scale/1` and `esa_readout_scale/1`.
The example scalar is initialized to one and stored in the existing addon
namespace. Base initialization is loaded before addon attachment; normal Session
checkpoints include addon parameters and their optimizer state.

`Modern2DFormalTrainingProfile(purpose="configured-screen", ...)` supports
explicit horizons and composed recipes for bounded CPU synthetic verification.
It is **not a native release**. The old formal context rejects this profile
before payload loading. Historical 60-pass reference contracts keep their
original constraints. No old qualification or canceled campaign is reused as
authority for a new experiment.

## Indexed shard lifetime

Whole-file hash verification is retained per process and file stat identity,
independent of a two-entry mmap handle LRU. Eviction does not invalidate an
unchanged file's verification. A changed stat identity requires another hash;
changes while opening are rejected. Serialized worker state contains neither
verified-file records nor open handles, so each spawned worker verifies its own
files. Graph outputs remain clones and PyG per-graph memoization is cleared.

This assumes accepted immutable storage; it is not a guarantee against an
adversary changing file contents while preserving all filesystem identity fields.
It does not remove initial full content validation or prove remote throughput.

New configured feature bindings exclude model-width parameters from transform
configuration. ESA source identity conservatively includes `esa_native.py`,
because that file still owns tokenization and padding/incidence masks. Legacy
bindings keep their original full source/config closure.

## Remaining integration

The ordinary scientific-input compiler still supports K1/3 and GPTrans/2 only.
Configured Modern2D publication, resource authority and acceptance-plan wiring
are unfinished and deliberately blocked, not silently mapped to the historical
campaign adapter. Native qualification, GPU training, throughput, scientific
comparison and desktop migration are not certified by CPU checks.

## Verification

Focused coverage lives in `test_modern2d_addon_session.py`,
`test_indexed_shard_lifetime.py` and the existing feature-binding tests.
On 2026-10-19, Luna executed the 12-module CPU batch: 227 passed and one failed
in 638.24 seconds. The failure was an unnecessary generalization of the legacy
native data-validation seam (`SimpleNamespace` lacked `train_rows`). The main
agent restored that method's original 100K behavior, without weakening tests.
The final-source rerun of formal release, formal training, addon Session and
shard lifetime suites passed all 88 tests in 22.35 seconds. These overlap with
the initial suite and are not added to its test count. The entire 12-module
batch was not rerun after the fix.

Original logs/JUnit and final rerun logs/JUnit are retained under
`E:/MolGap/artifacts/modern2d-addon-loader-tests-2026-10-19/`.
Warnings concerned CPU pin-memory, pkg_resources deprecation and JUnit
record_property compatibility. The main agent authored all implementation and
test changes; Luna only executed tests. No remote compute was used or released.

The bounded addon/loader CPU implementation is verified. The full normal-entry
Modern2D workflow remains **INCOMPLETE**, as described above.
