# Mount-resolution correction and prevention - 2026-10-13 JST

## Cause

Attempt1 used a relative manifest path under `/kaggle/input`, assuming a flat
dataset mount. Both workers rejected the missing joined path before graph or
label loading. Remote metadata named the correct dataset; its exact physical
location was not recorded, so a nested mount is an explanation, not an observed
fact of attempt1. The verified4790-byte published manifest has shard paths
such as `train/train_shard_0000.pt`, relative to its own dataset directory.
The shared loader also joined these paths to the general input root, which
would have failed after finding a manifest in a child directory.

Preparation checked source/archive/init transport and declared dataset names.
The dataset test copied a cache root directly to another root; it never put
that cache inside the multi-dataset filesystem the bootstrap passes. The
source-only release check therefore missed a real loader/bootstrap boundary.
This was an infrastructure defect plus missing integration coverage in the
preparation I used; successful source checks were insufficient for release.

## Reused correction

The shared loader now selects a missing location only by a unique frozen
manifest SHA and bases shard paths on that manifest's parent. An existing
wrong manifest, missing match or duplicate match fails closed. No recipe,
reference, split, role, target, initialization or optimizer is substituted.

The existing `check-release` and accelerator submitter now require dataset
metadata bindings for final shared-family releases and probe the actual frozen
resolver on flat/nested layouts before POST. Source-only preparation still
does not claim remote qualification. Kaggle bootstrap records exact resolution,
checks selected file presence and emits `INPUT_RESOLVED` before the native
qualification barrier. The owning trainer continues to hash and validate all
selected shards before decoding and training.

Synthetic tests cover both layouts through the loader, including shard-relative
paths, missing/wrong/duplicate identities and unopened unselected development.
Release tests cover absent/wrong bindings and a broken packaged resolver.
These safeguards prevent the specific unchecked path assumptions. Platform
mount failures, dependency changes and native qualification can still fail;
they must be reconciled from durable evidence rather than silently retried.
