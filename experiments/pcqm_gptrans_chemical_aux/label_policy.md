# Auxiliary label repair policy, 2026-09-30

The user authorized implementing the label-adapter repair and one final combined
test session. This record defines the new explicit capability; it does not
rewrite the failed strict cache attempt in `cache_decision.md`, release training,
or assert complete PCQM coverage. Future execution still needs prospective
records and pinned per-arm source, cache, recipe and comparison identities.

## Separate the required components

Descriptor-only and fingerprint-only caches contain only their required arrays.
The trainer rejects a cache lacking a component with a positive objective weight.
Inactive auxiliary tasks do not require labels or execute their loss. Legacy
cache v1, strict parsing and reject-on-missing defaults remain supported.

Explicit `pcqm_topology` parsing calls the existing PCQM feature-screen parser.
It retains the accepted topology fallback without constructing geometry. Graph
acceptance does not imply descriptor validity. Fingerprint-only preparation
does not import or construct the Descriptastorus generator.

## Explicit descriptor missingness

`mask_nonfinite` uses the library's descriptor functions and exact normalization
CDFs through an instance-local calculation hook. A raw calculation exception,
missing/nonfinite raw value, unavailable CDF, normalization exception or
nonfinite normalized value produces a missing cell. This prevents the upstream
normalizer's exception-to-0.0 fallback from looking like a valid observation.
Each descriptor is calculated once; library globals and the legacy strict
generator are unchanged. The encoder identity records
`raw-and-cdf-failure-to-nan-v1` in the hashed cache manifest.

The cache records every missing source index, descriptor column index and
column name, and retains a boolean validity mask with all original training
rows. Whole-generator failure flags still reject the row.

Missing cells have zero storage placeholders, never zero-valued observed labels.
The descriptor MSE averages only valid cells. A batch with no valid descriptor
cells has a connected zero auxiliary loss. Gap supervision uses every original
row; fingerprint loss is independent. No target, graph membership, optimizer,
precision, inference graph or exposure contract is changed by this adapter.

Cache v2 binds component selection, parser and missingness policy. The loss
fingerprint binds that policy in addition to the objective configuration; raw
cache hashes remain in addon/checkpoint/runtime identity. New masked objectives
cannot be represented as an unchanged strict-reference loss.

## Stop cheaply and retain failure evidence

Validate JSONL structure and exact role order before computing descriptors.
The owning CLI checks known exceptional PCQM train indices first; successful
arrays are always emitted in the original pinned role order. By default that
CLI stops at the first failed label and atomically retains an unaccepted
manifest with processed counts and an explicitly incomplete failure inventory.
`--collect-all-failures` deliberately requests an exhaustive diagnostic instead.

Source/role hash failures remain immediate input rejection. Structural input
failures and label failures do not publish labels.npz or an accepted cache.
This is failure recording, not an additional RML framework or scientific gate.

## Owning CLI examples for a later authorized cache attempt

Use `build_labels.py` with the existing authenticated export and role hashes:

- Fingerprint arm: `--components fingerprints --parse-policy pcqm_topology`.
- Descriptor arm: `--components descriptors --parse-policy pcqm_topology
  --descriptor-missing-policy mask_nonfinite`.

Do not reuse the interrupted cache, silently upgrade old manifests, or rerun
the full 100K build merely to verify implementation. The final repair tests use
small synthetic SMILES/cache/tensor fixtures, including the known Si string;
no full model, GPU, protected role or remote execution is involved.

The executed test result and the subsequent untested correction are recorded
in [repair verification](repair_verification.md). Implementation availability
does not establish full-cache qualification or remote readiness.
