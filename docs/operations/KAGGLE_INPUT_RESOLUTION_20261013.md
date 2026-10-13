# Kaggle pinned input correction - 2026-10-13 JST

Reviewed reusable correction from owning experiment commit `b7c137b2` on
`codex/exp/k1-mean2-single-ema-500k-20261013`. Integration includes shared
data resolution, release/submitter checks, platform observation and synthetic
tests only. The active experiment's implementation wrapper, prospective
records, failed attempt and scientific decision remain on its owner.

## Gap and correction

The bootstrap passed the multi-dataset `/kaggle/input` root to a recipe assuming
one flat dataset path. Shared loading also based shard paths on that general
root rather than the manifest directory. Source-only checks did not cover
that filesystem boundary. Native preflight correctly stopped both formal arms.
The actual failed worker mount location was not retained; nested layout was
not proven for that attempt.

`family_training_data.resolve_training_manifest` reuses frozen-hash metadata
identity to locate a missing manifest uniquely. An existing wrong manifest,
missing match or multiple matches fails. Shards are resolved relative to the
manifest, then undergo the existing role/hash/row/target checks. Final shared
family `check-release` requires `dataset_manifests` / `--dataset-manifest`
bindings and probes the packaged resolver on flat/nested mounts. The
accelerator adapter repeats this gate before POST. Startup retains exact
resolved paths, checks selected file presence and prints `INPUT_RESOLVED`
and phase transitions. Native qualification remains required before training.

## Verification and limits

gpt-6-luna/max used project Python, owning src and hidden GPUs. Four suites:
family_training_data, release_inputs, kaggle_bootstrap and experiment_cli.
First run:187 passed,5 failed,73.90s. After an ordinary-JSON compatibility
correction and synthetic package-fixture correction, only the five failed
tests were repeated:5 passed in17.94s. All192 distinct cases passed. One
PyG deprecation warning; no real molecular-role read or remote execution.

These tests establish the corrected path boundary and fail-closed metadata
gate. They do not guarantee remote platform mounting, dependency availability,
native repeat/resume qualification, training completion or scientific benefit.
The corrected remote attempt receives separate immutable source and receipts;
historical attempts are not changed by this integration.
