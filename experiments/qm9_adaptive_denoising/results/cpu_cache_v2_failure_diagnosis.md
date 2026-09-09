# CPU cache preparation v2 failure diagnosis

Kaggle2 kernel `kaseichou/molgap-qm9-adaptive-denoising-cache-prep`, version 2,
passed the repaired source-identity check and entered ETKDG cache construction.
It was then cancelled after every process worker failed while importing
`rdkit.Chem.AllChem`.

## Cause

The frozen `rdkit==2023.9.6` wheel was compiled against the NumPy 1.x C ABI,
while the current Kaggle Python 3.12 CPU image supplied NumPy 2.0.2. The remote
traceback explicitly reported that the module compiled with NumPy 1.x cannot
run under NumPy 2.0.2. This is an environment-compatibility failure, not an
ETKDG molecule failure or a model result.

The retrieved working output contains only incomplete raw/preprocessed source
material. It contains no cache `manifest.json`, completed shard set, aggregate
SHA, acceptance output, or scientific metric.

## Repair boundary

The CPU entry point now installs `numpy==1.26.4` before `rdkit==2023.9.6`.
Static contract tests freeze that ordering. No source molecule, role, split,
feature, ETKDG setting, model, target, seed, batch size, precision, optimizer,
schedule, or gate changed. Version 2 consumed no GPU time.
