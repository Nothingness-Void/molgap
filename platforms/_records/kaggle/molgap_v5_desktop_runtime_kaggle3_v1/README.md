# Kaggle3 V5 Desktop Runtime Evidence

Private dataset `nvoid912/molgap-v5-desktop-runtime` installs the reusable
desktop runtime layer for Kaggle3. It is bound to source commit
`24d9b13fad48bab1d0a8a92fec2b0b88b57224c9` and contains 147 tracked source
and contract files.

Kaggle expands the deterministic source archive into `molgap_runtime/`.
`acceptance.json` verifies every expanded path, size, and SHA256 against
`SOURCE_FILES.json`; the runtime manifest and sidecars are also checked. The
dataset is private, contains no credentials or protected benchmark roles, and
does not itself select a scientific contract or authorize training.

Kaggle records the source-only package license as `unknown`; keep the dataset
private and use it only for this repository's remote execution until project
licensing is explicitly settled.
