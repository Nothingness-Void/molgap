# Product recovery and bond-supported triplet routing

## Evidence and diagnosis (2026-10-13)

The [preceding terminal interpretation](../pcqm_gptrans_triplet_followup_100k/gpu/results/interpretation.md)
separates two outcomes. Branch drop completed at0.1416272528eV versus the
immutable aggregate's0.1393084501eV. At matched later epochs both training and
development error were higher; final live and EMA errors were similarly poor.
There is no evidence that merely extending training or changing EMA rescues
that intervention. Close exact whole-molecule branch drop; do not repeat it.

Product failed before forward/optimization. Its composite initialization guard
did not name the failed clause. Local torch2.7.1 accepted the frozen file while
the remote companion used torch2.11.0. Cross-runtime regeneration of new random
Linear tensors is an avoidable portability dependency, but the retained log
does not prove that particular clause caused the failure.
[PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
does not guarantee reproducibility across releases/platforms. Qualification
therefore checks pinned retained bytes, shapes/dtypes, parent identity,
parameter count and finiteness; it never substitutes new random values.

The [TGT method, equations5/6](https://arxiv.org/html/2402.04538v2#S3.SS1)
factorizes aggregation by dropping junction dependence. Its weight-level
triplet dropout is not the failed whole-molecule branch drop. The following
bond-supported extension is our hypothesis, not a reproduction of TGT's
geometry supervision or a published guarantee of improvement.

## Two independent interventions

Reference: the accepted triplet-aggregate terminal bundle retained under
`../pcqm_gptrans_triplet_followup_100k/reference/reference_bundle.json`.
No baseline rerun. Both arms retain its exact parent initialization tensors.

- A: channelwise relation product,5,885,185parameters. Recover the scientifically
  unevaluated product using the **exact original**5,885,185initial values and
  forward algebra. The new version changes qualification/provenance, not model
  capacity or scientific recipe. No trained checkpoint or warm-start is used.
- B: bond-supported aggregate,5,880,961parameters. Only the value-pair support
  becomes real directed bonds, self and virtual-node links. Normalize the
  third-pair probabilities over that support, then retain the sigmoid gate.
  Both directions and all valid output pairs, including virtual readout, remain.
  This restores a topology-dependent junction constraint without allocating a
  cubic attention tensor; dense contractions still have cubic arithmetic.
  Denominators clamp at1e-12. No new trainable tensors or forward RNG.

The rationale for B is targeted rather than proven: accepted real-bond and
triplet communication gains justify asking whether all-pair value mixing
dilutes useful chemical relations. Prior source diagnostics did **not** establish
broad collapse, and this experiment does not assert otherwise. Failure could
mean that nonbonded values were useful, normalization increased sensitivity,
or a one-seed reused development-role result was noisy.

## Immutable matched contract and stop gates

Fixed cross-platform100K asset, train[0,100000), internal-dev[100000,150000).
Pure2D OGB atom9/bond3/SPD20, direct normalized GapL1, unchanged train-only
transform. Seed42, FP32/noTF32, physicalBS128, accumulation1/drop_last32.
AdamW lr0.001/wd0.05/single-group/foreach=false/fused=false, clip1,
60passes,4warmup/cosine to1e-6, EMA0.999 each step, min-EMA-dev selection.
Exactly46,860steps/5,998,080presentations. No auxiliary loss or new cache.

One T4x2 notebook: one arm/model/RNG/optimizer/checkpoint tree per isolated
device. Both remote optimizer-inclusive preflights must pass **before either
formal training stream starts**. Initialization qualification writes named
checks. Preserve native atomic selected/last/periodic checkpoints and traces.
Estimate4–5wall hours/8–10allocatedT4hours; ceiling8wall/16allocated hours.
No automatic grid, seed expansion, scale training or failed-arm retry.

The frozen0.001eV material gain, paired-row positive interval and native
acceptance may nominate a single-seed follow-up only; row bootstrap is not
training-seed confirmation or full-scale superiority. Every completed arm
must independently pass strict comparison and actual Replay admission. A
preflight failure is infrastructure-only, never fabricated training Replay.

## Deferred audit and closure

Before compute release, package `gpu/post100k_audit_action.json` with its SHA
in the source inventory. Training acceptance precedes any separately bound
NO_TRAIN audit. Use all fixed500K internal-dev[500000,550000), not random10K;
unchanged100K transform, FP32/noTF32/BS128, reproduction tolerances MAE0.0001
and max-prediction0.001eV. Bind actual accepted checkpoint/prediction hashes,
labels only after saved predictions, atomic5000-row chunks, roles/nativecost
and linked NO_TRAIN RML. Missing matching loader/reference/full-role hashes
blocks that stage, not training acceptance. The monitor cannot launch it.
Official validation/test-dev/test-challenge, desktop jobs, IMS/SCNet and500K
training are forbidden. Existing Luna B handles exact-job30minute silent
monitoring and terminal handoff to existing A; no new chat/cron.
