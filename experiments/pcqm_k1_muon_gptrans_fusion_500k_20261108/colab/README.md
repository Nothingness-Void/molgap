# Colab notebook preparation — 2026-11-08

Local notebook only; no Colab connection, preflight, data read or training was
performed while building it. The prior NO_TRAIN fusion closure is unchanged.

Export: `D:/下载/MolGap-Colab-Full-20261108/MolGap_K1_Muon_GPTrans_Full_A100.ipynb`.
The notebook embeds the exact source-only archive from desktop5f55a7b5 and the
accepted canonical full-train manifest7d358a67. No second source upload, model
checkpoint, credential, geometry payload or full-output archive is included.

## Two modes

- `prepare` (default, CPU runtime): stage only the68accepted TRAIN topology
  shards (10,681,142,468bytes), validate with the existing full dataset loader,
  and write actual full-train mean/sample-std to durable Drive. This observation
  supplies a later full recipe freeze; it is not a scientific training result.
- `train` (one A100): consume two independently frozen single-arm Colab
  `prepare-family-training` bundles, then invoke the existing
  `molgap.experiment_training_worker` preflight/train phases sequentially.
  Each recipe must be full-train/final-only with epoch logging. K1 must use
  original single-forward plus batched Muon; GPTrans must use the accepted
  degree-bond-local/pair-update-norm mode. Dataset, step budget, optimizer,
  scheduler and initialization remain owned by each frozen bundle.

Outputs/checkpoints stay on Drive; input shards use local SSD. A cooperative
SIGTERM before the notebook session ceiling asks the original worker to retain
PAUSED state. Recovery invokes its existing actual-state preflight. Normal
completion and cell errors call Colab runtime.unassign; a final manual release
cell is also present. Actual cloud release has not been exercised locally.

## Inputs still needed before formal training

The Drive location of the accepted full cache has not been inspected. Set
FULL_CACHE_ON_DRIVE to its real root;500K mirrors are not full data. Only the
accepted TRAIN topology files are needed, not the25GB combined topology/3D
store. The existing accepted manifest is a canonical JSON digest, so the
notebook stages those exact canonical bytes without changing any shard digest.

Modern Colab full prepared bundles for these two new recipes do not yet exist.
Full normalization, step/exposure budget, final-live/final-ema policy and budget
authority must be frozen through the existing preparation owner after the CPU
observation. They belong under RELEASE_ROOT/k1_muon and
RELEASE_ROOT/gptrans_pairnorm; the train mode rejects missing or500K inputs.
No historical full checkpoint or old full trainer substitutes for this step.

The source-only snapshot embeds all existing MolGap Python modules to avoid
the earlier missing-pickle-module incident. Training still loads the exact
per-arm packaged source, verified by the existing package owner. This notebook
does not install a new model, training loop, RML schema or verifier.

Build with the primary project Python and this build_notebook.py. Validation is
JSON/nbformat structure plus Python AST on every code cell, decoded source SHA
and accepted manifest SHA; no GPU, environment installation or cloud run.
