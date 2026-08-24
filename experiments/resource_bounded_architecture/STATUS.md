# Live Status

Updated: 2026-08-25 07:52 JST.

The 2026-08-25 07:52 JST sequential Kaggle inspection found no pending or
running MolGap kernel. All 10 architecture-tournament kernels remain terminal
`COMPLETE`; among the 14 historical kernels whose titles still say `ACTIVE`,
13 are `COMPLETE` and the already-diagnosed round-07 acquisition remains
`ERROR`. No repaired-2M EdgeState kernel was submitted because the complete
immutable full input and measured one-epoch timing projection required by
`ROADMAP.md` do not yet exist. Two unrelated historical acquisition kernels
still report `ERROR`; their durable partial outputs and logs were retrieved for
diagnosis under
`platforms/_records/kaggle/training/molgap_kaggle_error_inspection_20260825_01/`.

## Paired Screen

The fixed PubChemQC 100K GPS9 versus RWSE16 Structural GPS screen completed on
Kaggle and passed local read-only acceptance.

| Variant | Kaggle kernel | Version | State |
|---|---|---:|---|
| GPS9-192 control, seeds 42/43/44 | `nothingnessvoid/molgap-pc100k-gps9-control-20260824` | 4 | Accepted |
| RWSE16 Structural GPS9-192, seeds 42/43/44 | `nothingnessvoid/molgap-pc100k-structural-gps9-20260824` | 4 | Accepted |

All six runs retain best and last checkpoints, metrics, and 9,997 aligned test
predictions. The three seeds improved validation and test MAE in one direction,
and the predeclared `0.001 eV` gate passed. Exact metrics and the bounded
promotion decision are in `decision.md`; machine acceptance is in
`results/paired_100k_screen/acceptance.json`.

Immutable private inputs:

- graph/split dataset:
  `nothingnessvoid/molgap-pc100k-rwse-screen-20260824`;
- source runtime dataset:
  `nothingnessvoid/molgap-pc100k-architecture-runtime-20260824`;
- base graph SHA256:
  `5c348c28c8f75f09d6072ebf88de28f8513feec6e7a20c8439edc12d4b18d936`;
- RWSE graph SHA256:
  `71d61923ea008b02eb902003c73eaf9aee9f6ff488be5f88482f0e11d700e017`;
- split SHA256:
  `1e6707274dd8465cfe9d96a808064372af705c4a9e4b8d20532ae6fff2cdcf05`.

Versions 1 through 3 stopped before training and remain incident provenance.
Version 4 used a writable runtime copy, P100-compatible PyTorch 2.7.1/CUDA
12.6, exact 120K preflight, and durable per-seed artifacts. SCNet was not used
because the replacement account had no authorized Slurm partition.

The accepted result authorizes one repaired-2M Structural GPS scale-up. It does
not authorize production promotion or opening common/OOD/P8-hard before that
standalone model is complete. Sealed data remains locked.

## Gap And RWSE Round

Two private kernels completed on 2026-08-24. Both mounted the accepted
graph/split dataset and a new immutable runtime dataset; neither overwrote the
accepted paired-screen kernels.

| Question | Kaggle kernel | State |
|---|---|---|
| Does Gap-only supervision beat the old Structural GPS Gap head? | `nothingnessvoid/molgap-pc100k-structural-gap-only-r1` | Accepted; hypothesis rejected |
| Does normalized, learnably gated RWSE beat Gap-only Structural GPS? | `nothingnessvoid/molgap-pc100k-normalized-rwse-gap-r1` | Accepted; local gate passed, no promotion |

The runtime dataset is
`nothingnessvoid/molgap-pc100k-gap-rwse-runtime-r1`; its source archive SHA256
is `27f345d2755fbf9fd2a4d71c901f5225505e38f6e3b893af6323f3449b390b26`.
Each kernel ran seeds 42/43/44, wrote an epoch checkpoint, and required strict
one-column prediction acceptance. All six runs and their 9,997-row aligned test
predictions passed read-only acceptance. Gap-only regressed against the old
three-output Gap head in every seed. Normalized/gated RWSE improved Gap-only in
every validation seed but remained worse than the old three-output model, so it
did not receive repaired-2M authorization. Exact metrics and the decision are
in `results/gap_rwse_100k_screen/decision.md`; machine acceptance is in
`results/gap_rwse_100k_screen/acceptance.json`.

The conservative 2D+3D implementation passed local exact-identity, bounded
correction, fallback-selection, compilation, and resume tests. Its immutable
external payload contains 998 OOD and 975 P8-hard ETKDG-valid rows and passed
identity, shape, finite-value, and SHA256 acceptance. IMS CPU job
`1327453.ccpbs1` assembled the compact repaired-2M training payload under a new
handoff directory without requesting a GPU or changing the running PairGPS job.
The payload contains 198,932 rows and a 160,057/19,501/19,374 scaffold-disjoint
train/validation/test split. The scheduler emitted a two-core cgroup memory
warning after both atomic outputs were written; remote SHA256 and complete
local tensor acceptance passed, so the artifact was retained without rerun.
The reusable PBS template now requests four cores for the corresponding memory
allowance. The Drive-backed A100 notebook, wheel, manifests, and four inputs are
packaged; model training remains unstarted.

## Gated Structural Feasibility

The three-target edge-aware GatedGCN plus RWSE16 Structural GPS feasibility
screen was submitted as a new private Kaggle kernel:

- kernel: `nothingnessvoid/molgap-pc100k-gated-structural-seed42-r1`;
- runtime dataset: `nothingnessvoid/molgap-pc100k-gated-runtime-r1`;
- runtime archive SHA256:
  `79988b4804a0c2efa0f157811b959e1e3c374646f2526e51626ed19a2b4531e0`;
- immutable graph/split dataset:
  `nothingnessvoid/molgap-pc100k-rwse-screen-20260824`;
- scope: three targets, 40 epochs with per-epoch resume;
- all three seeds: complete and locally accepted;
- confirmation kernels:
  `nothingnessvoid/molgap-pc100k-gated-structural-seed43-r1` and
  `nothingnessvoid/molgap-pc100k-gated-structural-seed44-r1`.

The exact batch-256 local accelerator preflight was finite and used about
`0.84 GiB` peak allocated memory. A warmed forward/backward step took `1.47x`
the accepted Structural GPS step, projecting below the bounded screen limit.
Seed 42 passed its gate: validation average MAE improved from `0.134945` to
`0.127606 eV`, test average MAE improved from `0.136832` to `0.131100 eV`, all
three targets improved, and training took `2730.93 s`. The accepted report is
`results/gated_structural_100k_seed42/acceptance.json`. This authorized the two
confirmation seeds but did not authorize repaired-2M scale-up. The final
three-seed gate failed because seed 43 regressed despite positive seed42/44,
mean, and equal-ensemble results. Exact evidence and disposition are in
`results/gated_structural_100k_multiseed/decision.md`.

## Persistent Edge-State Structural GPS

The bounded feasibility and two confirmation seeds completed and passed strict
local acceptance:

- kernel: `nothingnessvoid/molgap-pc100k-edge-state-structural-seed42-r1`;
- runtime dataset: `nothingnessvoid/molgap-pc100k-edge-state-runtime-r1`;
- runtime archive SHA256:
  `9719e7ba4442939b50362ff852307d90379aa617943efbb42642646a8af68d47`;
- immutable graph/split dataset:
  `nothingnessvoid/molgap-pc100k-rwse-screen-20260824`;
- scope: seed 42 only, three targets, 40 epochs with per-epoch resume;
- accepted seed-42 decision:
  `results/edge_state_100k_seed42/decision.md`;
- confirmation kernels:
  `nothingnessvoid/molgap-pc100k-edge-state-structural-seed43-r1` and
  `nothingnessvoid/molgap-pc100k-edge-state-structural-seed44-r1`, both terminal
  `COMPLETE` and strictly accepted from private version-1 outputs.

The exact local batch-256 preflight completed with finite forward/backward
values. The candidate used `0.939 GiB` peak allocated accelerator memory,
`4,739,267` parameters, and a `0.0591 s` warmed step on the local comparison
device. Validation improved in seeds 42/43/44 by `0.004943`, `0.008009`, and
`0.007240 eV`; the mean gain was `0.006731 eV`, so the exact three-seed gate
passed. The accepted decision is
`results/edge_state_100k_multiseed/decision.md`. One repaired-2M EdgeState run
is authorized after its complete immutable input and timing projection pass;
no full-scale job has been submitted.
