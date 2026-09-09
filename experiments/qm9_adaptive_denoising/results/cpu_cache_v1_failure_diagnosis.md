# CPU cache preparation v1 failure diagnosis

Kaggle2 kernel `kaseichou/molgap-qm9-adaptive-denoising-cache-prep`, version 1,
terminated before importing the cache builder. The terminal traceback is
preserved under ignored platform evidence storage; its compact handoff marker
is retained beside the experiment protocol.

## Cause

`SOURCE_TREE_SHA256.txt` was generated on Windows by sorting `Path` objects.
That ordering is case-insensitive. Kaggle recomputed the digest on Linux with
case-sensitive `Path` ordering. The packaged tree contains both
`archive/README.md` and lower-case names, so the aggregate order diverged even
though the 130 relative paths, file lengths, and individual SHA-256 values
matched exactly after downloading the published source dataset.

- Windows-native aggregate: `760c133e1f67234b57e9ff52d1ed8ba0b72c50bc6e7b0e2edd762cb41cabd8f7`
- Canonical POSIX-relative aggregate: `7afd53922bb414e1ffda9c89b5d0e19fa2708edf2455e808243ffe032ded089e`
- Terminal log SHA-256: `b3c3b9cdd8308c263385a1b18f8941619e8d9c8671e194bfdb610afaab6d19f0`

## Repair boundary

The packager and both remote entry points now sort explicit POSIX relative-path
strings. A mixed-case cross-platform regression test freezes this behavior.
No data role, molecule, split, feature, ETKDG setting, model, target, seed,
batch size, precision, optimizer, schedule, or promotion gate changed. Version
1 consumed no GPU time and produced no scientific result.
