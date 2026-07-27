# PubChemQC 100K architecture — completed remote rounds

Dated record of the frozen scaffold-disjoint 100K/10K/9,997 screen. Nothing here
promotes a production model.

## Pure-2D gate

Jobs `709051`-`709054` completed. GPS11-160 has the best pure-2D validation and
test average MAE and the best test Gap MAE, so it advances with GPS7/GPS9 controls
to the two-SchNet fusion screen.
Acceptance: `remote_acceptance.json`.

## Second-conformer preparation (version 3 accepted)

Four bounded CPU kernels, `nothingnessvoid/molgap-pc100k-conformer-r0` through
`r3`. Version 1 failed before data processing because Kaggle did not include the
sidecar `variant.json`; version 2 embedded the shard identity but exposed that
the CPU image lacked RDKit; version 3 installs the pinned RDKit dependency and
embeds the shard identity. Local acceptance loaded and hashed all 24 graph parts:
119,602 of 120,000 molecules succeeded, all retained `source_idx` values are
unique, and labels/coordinates are finite.

Immutable split input:
`nothingnessvoid/molgap-pubchemqc100k-arch-split-20260725`.
Accepted cache: `nothingnessvoid/molgap-pc100k-second-conformer-v3-20260725`,
**version 2** — version 1 is incomplete because the CLI skipped nested
directories and must never be mounted for training.
Exact counts: `remote_acceptance.json`.

## Lightweight SchNet branches

The SchNet contract is the lightweight `176/160/6` architecture for both
conformer branches; the legacy `192/192/6` SchNet is explicitly forbidden.

Kernels `nothingnessvoid/molgap-pc100k-light-schnet-primary` and
`...-augmented` version 1 both failed before epoch 0 because Kaggle assigned P100
GPUs while stock `torch 2.10.0+cu128` omits `sm_60`. Version 2 conditionally
installs the previously validated `torch 2.7.1+cu126` compatibility runtime; both
kernels completed and both checkpoints plus embedding payloads passed strict
acceptance. The augmented model is materially stronger than the primary-only
model even under one-view inference.

## Fusion screen (Precision architecture selected)

The three-seed frozen Fusion screen selected the strict two-SchNet-pass Precision
architecture: GPS9 + GPS11-160 + primary SchNet + two-conformer-trained augmented
SchNet, with both SchNets evaluated on one primary conformer. Test average/Gap MAE
`0.138046/0.165819 eV`, improving pure GPS11-160 by `0.004424/0.005221 eV`. A
third SchNet forward improves only `0.000699/0.000754 eV` and is rejected on cost.
Decision: `route_b_fusion_decision.md`.

## Head A/B and correction bound

A three-seed head A/B replaced the shared gated-sum bottleneck with a GPS11-160
identity path plus a bounded residual correction. A validation-only three-scale,
three-seed A/B selected a `+-0.10 eV` bound. The frozen head reaches test
average/Gap `0.134463/0.160809 eV`, improving the original gated head by
`0.003583/0.005010 eV`. This is the retained full-scale Fusion protocol;
external common/OOD/P8-hard evidence is still required.
Decisions: `route_b_head_ab_decision.md`, `route_b_residual_scale_decision.md`.
Manifest: `experiment_manifest.json`.
