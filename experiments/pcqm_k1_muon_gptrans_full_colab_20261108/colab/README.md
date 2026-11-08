# Full-training Colab delivery

The user starts the A100 runtime. Local preparation does not execute training.
Both independent packages were produced by `prepare-family-training`, with
`SOURCE_INPUTS_PREPARED`, no release errors and `submitted=false`.
Frozen source: `e17758f8f42892b4b0163669b11092ac34e92112`.

## Run

The same notebook is staged privately in
[Google Drive](https://drive.google.com/file/d/14c0zOXY2XvlV6bYtWwKKfqEbJdnEhshW/view).

1. Upload `D:/下载/MolGap-Colab-Full-20261108/MolGap_K1_Muon_GPTrans_Full_A100.ipynb`
   to Colab, select one A100 and run all cells. Authorize the Drive mount.
2. The notebook reads the already staged private `MyDrive/MolGap/V5` assets,
   checks transport hashes, extracts the two frozen prepared packages, and stages
   the 68 TRAIN topology files to SSD with original SHA256 verification.
3. It calls the existing worker's native preflight and train phases, first
   `k1_muon`, then `gptrans_pairnorm`. Reopen this same notebook and run all cells
   to resume its durable state. Do not change the run directory or recipes.

Per model: 3,378,606 TRAIN rows, 60 full passes, 1,583,700 optimizer updates,
202,713,600 sample presentations, BS128/drop-last and FP32. K1 uses batched
Muon and final-live; GPTrans uses BondLocal+PairNorm and final-EMA. No internal
development role or official validation/test input is mounted. TRAIN MAE is
logged each epoch; it is not a benchmark evaluation.

The notebook sends cooperative SIGTERM before its ten-hour session cap, then
releases the runtime after the shared runner saves PAUSED state. It restores
optimizer, scheduler, EMA, RNG, sampler and progress from the same checkpoint.
This is a multi-session fixed plan, not a promise to finish both models in one
session. Actual A100 throughput is measured at execution, not extrapolated from T4.
Normal completion, controlled pause and cell errors invoke Colab's
`runtime.unassign()`. If stopping manually, disconnect and delete the runtime.

## Asset ownership

- Local exported notebook and the two transport ZIPs:
  `D:/下载/MolGap-Colab-Full-20261108/`.
- Original prepared directories:
  `D:/molgap-artifacts/colab-full-20261108/releases/{k1_muon,gptrans_pairnorm}`.
- Drive prepared ZIPs: `MolGap/V5/releases/k1-muon-gptrans-full-20261108/`.
- Drive TRAIN transport ZIPs:
  `MolGap/V5/fixed_data/full_train_v1/archives/`.
- Unique runtime outputs/checkpoints/native proofs:
  `MolGap/V5/runs/k1-muon-gptrans-full-20261108/`.
- `release_assets.json` pins the transport and scientific package identities.
  `data_archives.json` pins all 23 data ZIPs and their 68 original paths.
  `train_transfer_receipt.json` records the verified original topology transfer.

The Drive API rejects original shards larger than 100 MiB. Only the necessary
TRAIN topology was therefore compressed into 23 ZIPs below that transport limit;
no geometry, official-valid/test or raw source dataset was transferred. The
original data identity is unchanged. The notebook unpacks the TRAIN transport ZIPs; the existing training data
loader verifies the original manifest and every selected shard before use.
The notebook does not repeat that data hashing. Existing sharing settings are unchanged.

## Reuse and qualification

Scientific records use the existing `FamilyRecipe`, `build_family_recipe`,
`ExperimentSpec`, family bindings and prospective publication. Training,
native qualification, exact resume, atomic artifacts and standard epoch logs
remain owned by `experiment_training_worker` and the shared family runner.
The added notebook handles Drive/SSD transport, environment setup, sequential
launch, a bounded session signal and runtime release. Shared training/RML source
was not changed for this delivery. An already completed output is inspected
through the owner's completed-run path before it is skipped.

`build_notebook.py` reproduces the exported notebook from the frozen git source
and the two tracked asset indexes. Its code cells are syntax checked at build.
Local preparation and static verification do not replace the actual A100 native
preflight. Scientific results and measured costs await user-started training;
subsequent acceptance uses the existing family/V5/RML owners and retrieves only
the required retained artifacts.

## Simplified launcher revision

The notebook loads bootstrap source from the already verified K1 prepared
package. It embeds no Base64 source or manifest. Preflight stdout/stderr is
retained in a Drive `.log` beside its preflight directory and displayed before
checking the return code. The reported preflight exit 1 remains undiagnosed
until the worker traceback is captured; this change does not claim to fix its
unknown cause. No frozen scientific package or training state was changed.
