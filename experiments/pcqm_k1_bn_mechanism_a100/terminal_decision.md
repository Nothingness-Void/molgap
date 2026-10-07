# Accepted frozen BN mechanism diagnostic — 2026-10-08

Disposition: **NO_TRAIN**. Mechanical acceptance passed for the one returned
Colab A100 attempt. This is frozen-checkpoint inference, not a training run,
training replay qualification, model adoption or a full-data release.

Authority: [protocol](protocol.md), [acceptance](acceptance.json),
[machine-readable paired analysis](analysis.json), [attribution](attribution.md),
and the hash-bound returned artifacts in `results/attempt-001/`.
The original prospective decision remains intact in `decision.md`.

## Observations

MAEs below are recomputed from aligned predictions using float64 absolute
errors. They can differ slightly from the worker's float32 console reductions.

| Frozen BN state | Gap MAE (eV) |
|---|---:|
| Original epoch49 state | 0.104904078307 |
| Dropout off, one pass | 0.103658943601 |
| Dropout off, two passes | 0.103658944602 |
| Dropout on, one pass | 0.104506773977 |
| Dropout on, two passes | 0.104568311510 |

All cases use the same 16,384 training members for feature-only calibration
and the same historically selection-used 50K internal-development cohort.
Learned parameters and non-BN buffers remained unchanged. Eighteen BN modules
received 128 or 256 updates. Original state and first128 predictions were
restored exactly between cases. Baseline reconstruction max absolute error was
1.907349e-6 eV, within the frozen 1e-4 eV tolerance.

| Paired contrast, positive means first state has greater error | Difference (meV) | 95% row-bootstrap interval (meV) |
|---|---:|---:|
| Original minus dropout off, one pass | 1.245135 | [1.107808, 1.379105] |
| Dropout on minus off, one pass | 0.847830 | [0.727312, 0.969685] |
| Dropout on minus off, two passes | 0.909367 | [0.787190, 1.032088] |
| Dropout off, two minus one pass | 0.000001 | [-0.000002, 0.000004] |
| Dropout on, two minus one pass | 0.061538 | [0.056632, 0.066512] |

The existing paired bootstrap uses 1,000 draws, seed20261008. These intervals
describe row sampling, not training-seed variance. The four predeclared new
mechanism contrasts all have point differences below the 1meV nomination gate;
**zero of four qualifies for material mechanism nomination**. The original
clean-BN recovery reproduces an earlier observation on the same cohort, rather
than adding an independent generalization result.

## Decision and scope

Clean BN estimation improves this frozen predictor; dropout during buffer
estimation makes its clean-evaluation predictions worse. Retain this evidence
and the reusable calibration controls. Do not infer that training dropout harms
learned weights, remove training regularization, or adopt this calibrated state
from an exploratory, previously consumed development cohort.

Repeated clean passes are a near-null control under reset plus cumulative
averaging. They do not reproduce historical EMA updates while weights evolve.
Training-time double-forward accuracy, native T4 epoch bottlenecks, capacity and
exposure explanations remain unresolved by this diagnostic.

Use the already authorized 500K consistency question's original and fixed
clean-BN outputs as its planned discriminator. This closure authorizes no new
training, inference, successor or protected-role access.

## Custody and cost

The returned payload manifest matches the frozen local manifest byte-for-byte;
all completion-listed artifacts passed SHA256 verification. Frozen executable
source is retained in `frozen_source/`. Large input/checkpoint payload remains
in ignored staging and private Drive under the recorded identities.

The worker completed in 173.537 seconds on native A100-SXM4-40GB, approximately
0.048205 allocated A100 hours. This is process wall during allocation, not GPU
busy time or total Colab CCU billing. Setup wall is separately recorded; setup
device allocation, idle allocation and local staging duration remain unknown.
No optimizer, gradients or training objective were created. Official validation,
test, common and OOD roles were untouched.

The user confirmed runtime disconnection and deletion after retrieval. This is
user-confirmed release, not independent browser verification.

Git route: accepted diagnostic with reusable implementation, reviewed integration
into `molgap-desktop` under BRANCHES.md. Model recommendation is unchanged.
