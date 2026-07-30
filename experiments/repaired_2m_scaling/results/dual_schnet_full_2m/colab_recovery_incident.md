# Colab Primary Recovery Incident

## Impact

The Colab Primary recovery run consumed paid A100 time and reported a
scientifically invalid test average MAE of `0.623343 eV`. Its output is rejected
and must not be used for comparison, fusion, or model registration. No accepted
model was overwritten.

## Evidence

- Colab reported validation MAE `0.169136/0.202598 eV` at epochs 2/3 and
  selected `best_epoch=-1`.
- It then reported test HOMO/LUMO/Gap MAE
  `0.224299/0.836891/0.808839 eV`.
- The source recovery checkpoint SHA256 was
  `858f48dd8455e07c89d2631fe99dbc91d75ff326faaaa566e45b8fce0641320a`.
- A complete local replay of that source best state on all 198,925 test rows
  produced `0.119071/0.116855/0.160279 eV`, average `0.132068 eV`.
- The independently accepted IMS Primary model produced test average
  `0.120416 eV`; local replay reproduced it within `0.00004 eV`.

The Colab result therefore cannot represent the claimed source best state under
the accepted graph and split contract.

## Root-Cause Status

The exact low-level cause cannot be uniquely identified without the rejected
Drive output checkpoint, environment record, and local staged-cache hashes.
The previous notebook allowed three unresolved failure modes:

1. local Colab shards were reused when file sizes matched, without SHA256;
2. the recovery checkpoint was not evaluated before paid training;
3. final evaluation had no replay or validation/test divergence gate.

An old MolGap wheel could also be installed with a newer notebook because the
wheel itself had no required code hash.

## Corrective Controls

The replacement run is fail-closed:

- verify the acceptance JSON SHA256 and all 100 Drive shard SHA256 values;
- verify each copied local shard and compare the complete shard ledger;
- require the exact recovery checkpoint SHA256;
- use a new output directory so the rejected checkpoint cannot resume;
- record Python, PyTorch, CUDA, PyG, torch-cluster, GPU, and wheel hashes;
- run first/last-shard forward/backward preflight;
- replay the recovered state on full validation and test before epoch 0;
- evaluate the frozen test state twice and reject inconsistent results;
- reject excessive validation/test divergence;
- reject the first non-finite loss or gradient;
- record whether the selected model came from recovery or new training and hash
  the selected tensor state.

## Cloud Acceptance Rule

A remote encoder output is not complete merely because the training cell exits.
It is accepted only when immutable-input hashes, environment identity, recovery
baseline, final evaluation replay, selected-state identity, checkpoint hashes,
and an independent downstream acceptance record all pass.
