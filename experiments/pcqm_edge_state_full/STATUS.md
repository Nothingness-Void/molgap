# Remote Status

The IMS execution root is
`/lustre/home/users/sm2/chou/molgap-pcqm-edge-state-full`.

The official row stage completed with archive SHA256
`628ae612a3de752160929d93d1584a75257ae156e7b13d91b133d8db62e6d701`,
`3,378,606` train rows, `73,545` valid rows, and no materialized test SMILES.
The dependency chain is:

- `1344420.ccpbs1`: completed official train/valid row preparation;
- `1344421[].ccpbs1`: rejected before artifact creation because Torch tensor
  multiprocessing file descriptors failed on IMS;
- `1344428[].ccpbs1`: diagnostic single-process pass; it exposed 33 official
  SMILES requiring topology-only fallback and 24 rows containing nine rare elements;
- `1344441[].ccpbs1`: completed full-vocabulary topology-fallback graph array
  with 75 version-matched reports and zero failed rows;
- `1344442.ccpbs1`: completed strict acceptance of all `3,452,151` rows;
- `1344443.ccpbs1`: completed ten A100 epochs in `6.00 h`; epoch 9 was selected
  with official-valid Gap MAE `0.189450 eV`. Checkpoint, metrics, completion
  manifest, and aligned validation predictions passed hash retrieval.
- `1348205.ccpbs1`: frozen-model raw-SMILES test inference failed before
  producing NPZ files when Torch multiprocessing lost tensor file descriptors.
  The retrieved log is `results/test_inference_failure.log`. It was not
  resubmitted because the frozen checkpoint had already failed the accuracy
  gate.

The official test job consumes the frozen remote training completion manifest;
it does not reopen epoch selection.

Machine-readable submission provenance is in `results/submission.json`.
Accuracy root cause and the bounded repair gate are in `root_cause.md`.

## Rich-feature repair screen

The separate official-train-only screen uses the accepted paired input under
`feature_screen/graphs` and the fixed `100,000` train plus `10,000`
development rows. Its retrieved acceptance record is
`platforms/_records/ims/pcqm_feature_screen_20260826/graph_acceptance.json`
with SHA256
`faecf13321e373e76216e7cd6a6ab64e826d6983d2e595f419874e410d0bb3a4`.
The CUDA preflight passed for both legacy and OGB categorical schemas on an
A100, with finite gradients and the same acceptance SHA.

Legacy and OGB seeds 42, 43, and 44 completed. The paired acceptance passed;
its immutable result and decision are under `results/feature_screen/`.

The schedule screen reused the same accepted OGB graphs and compared
`collapse10` with `warmup20` for all three seeds. IMS jobs `1350938` through
`1350944` completed and the paired acceptance passed. Retrieved evidence is
under `platforms/_records/ims/pcqm_schedule_screen_20260827/`; the experiment
decision is `results/schedule_screen/decision.md`. Official validation and
test remained unread.

The immutable full OGB-rich cache then passed acceptance. IMS benchmark
`1352557.ccpbs1` measured `2245.9979 s/epoch`, projecting twenty epochs to
`12.4778 h`. The original 12-hour experiment budget rejected conditional job
`1352593.ccpbs1` before training. On 2026-08-27 the user explicitly authorized
one replacement submission with a 14-hour scheduler walltime and a resumable
13.5-hour training boundary. The original rejected timing record remains
immutable and hash-checked by the replacement job. Replacement job
`1355016.ccpbs1` completed all twenty epochs on 2026-08-28. Epoch 19 was
selected at official-validation Gap MAE `0.102063 eV`; the checkpoint,
resumable state, aligned 73,545-row validation predictions, metrics, and
completion-manifest hashes passed local acceptance. The decision and
machine-readable record are under `results/rich_full/`. Official test was not
read.
