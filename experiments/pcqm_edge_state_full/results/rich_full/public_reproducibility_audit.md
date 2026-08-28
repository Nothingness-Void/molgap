# Public Reproducibility Audit

On 2026-08-28, public repository head `ae00b44` was cloned from GitHub into a
new directory and installed into a new Python 3.10 virtual environment. The
installed `molgap` package resolved to that clone rather than to the private
MolGap workspace.

The clean environment passed dependency resolution, bytecode compilation, the
seven public tests, CLI startup, and PDF report reconstruction. The published
Release checkpoint matched SHA256
`7f09b3b0456a71cdb745a16ace6c7e4afe807afbac786d18a6f15a5e6e97da15`,
loaded strictly with all keys matched, and contained exactly `4,771,073`
parameters. A raw `CCO` SMILES completed OGB feature construction, RWSE16, and
a finite CPU model forward pass.

The same public code read the accepted official archive, reproduced all
`3,746,620` split identities, materialized exactly `3,378,606` train and
`73,545` validation rows without test SMILES, and built the real final source
shard of `9,214` validation graphs twice with identical hashes. The public CLI
then accepted both frozen official NPZ files with the recorded row counts,
float32 dtype, finite values, timing manifest, and SHA256 hashes.

One non-model defect was found: `report/build_report.py` used ReportLab without
declaring it. Public commit `ae00b44` added the optional `report` dependency and
documented the rebuild command. A second fresh clone of that final commit
passed the complete audit. The 12.14-hour full training was not rerun because
its accepted checkpoint, validation predictions, and completion evidence were
already immutable.

Machine-readable evidence is `public_reproducibility_audit.json`.
