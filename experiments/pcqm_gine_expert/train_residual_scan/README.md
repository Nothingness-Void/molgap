# PCQM4Mv2 Official-Train Residual Scan

## Purpose

Rank hard PCQM4Mv2 training molecules using the accepted equal
`control_a` + `repair_v2` pure-2D teacher. This is a candidate-data analysis,
not an OGB submission and not permission to train on official validation or
test rows.

## Cloud Execution

- SCNet job: `703665`, dependency `afterany:703653`.
- Input split: official `train`, 3,378,606 rows; official valid/test excluded.
- Input copies: team-share `molgap-data/pcqm4m-v2/`, with local/remote SHA256
  equality checked for `raw/data.csv.gz` and `split_dict.pt`.
- Output: team-share `molgap-results/phase8/pcqm4mv2_train_residual_scan/`.
- Durability: 136 independently readable 25K-row Parquet parts, one JSON report
  per part, atomic progress, final manifest, and a 200K residual-hard pool.
- Historical exclusion: canonical-SMILES deduplication against repair-v3 1.5M.

A bounded local preflight produced 49 parts before it was intentionally stopped
when the computer shutdown requirement was clarified. Those local parts are
not the authoritative run and are not inputs to job `703665`.

## Acceptance Gate

Accept only after all 136 part reports, row coverage, hashes, finite labels and
predictions, official split exclusion, historical deduplication, and final pool
manifest pass validation. PCQM4Mv2 provides Gap labels here; using selected rows
for three-target MolGap training requires either a masked Gap-only objective or
same-definition PubChemQC HOMO/LUMO recovery.
