# G1 decay recovery and pair-depth scaling: terminal decision

On 2026-10-02, Kaggle3 kernel `nvoid912/molgap-gptrans-g1-pair-scale-dual-s42`
ID 136742270, version 1, was independently verified COMPLETE. Both arms passed
saved-artifact acceptance and their own STRICT_CAUSAL comparison. No model was
executed locally. The accepted EMA0.999 reference was reused, not retrained.

## Endpoint evidence

| Arm | Saved prediction MAE (eV) | Candidate minus reference | Paired row-bootstrap 95% interval | Best epoch (zero-based) |
|---|---:|---:|---|---:|
| Frozen G1 EMA0.999 | 0.14423262914597987 | -- | -- | 58 |
| Bias/1D no-decay recovery | 0.1446980793374777 | +0.00046545019149780276 | [-0.00040855092525482174, +0.0013432114390134803] | 52 |
| Pair update divided by sqrt(12) | 0.14679757902562618 | +0.002564949879646301 | [+0.0016573998084068297, +0.003450411695957184] | 59 |

All endpoints were recomputed from 50,000 aligned saved internal-development
predictions. Physical BS128, FP32 without TF32, seed42, initialization and fixed
data identities were verified. Both arms had 5,246,817 parameters, 46,860 steps
and 5,998,080 sample presentations. The prospectively declared 0.003 eV promotion
gate belonged to this experiment, not a universal V5 threshold. Neither passed.

## Attribution and stopping

**Grouped decay:** exempting biases and 1D tensors did not establish a benefit.
The endpoint interval crossed zero, so this was lack of positive evidence, not
a stable proof of harm. EMA trailed the reference at epochs 9/19/29 by
0.006343/0.004687/0.002374 eV and recovered near parity late. Its final ten-epoch
EMA improvement was only 0.000040 eV. Observed preclip gradient norm fell from
3.043 to 0.497, and clip frequency from 87.96% to 2.43%. The repaired diagnostic
writer worked; there was no observed late numerical breakdown. Without matched
reference gradient diagnostics these measurements cannot prove which parameter
group caused the lack of improvement. The exact recovery hypothesis was closed.

**Pair depth scaling:** the recurrence was active and finite, but its early
advantage was not durable. At epochs 19/29 EMA gains were 0.000333/0.000160 eV;
at 39/49/59 they reversed to losses of 0.001694/0.002248/0.002556 eV. Compared
with the simultaneous grouped arm, final online training MAE was lower
(0.097317 vs 0.099071), yet EMA development error was higher. Thus a better
training fit or intermediate checkpoint did not establish better generalization.
Those arms also differed in optimizer grouping, so their head-to-head contrast
was descriptive; each strict claim used the frozen reference instead.

The scaled arm's first scheduled training batch at epoch59 had pair RMS 0.547
entering layer1 and 1.110 after layer12, with raw-update RMS 0.726/0.197 at the
first/last layer. This verified a functioning, finite relation stream. No baseline
pair-activation trace existed: neither baseline explosion nor the claim that
scaling specifically destroyed chemistry could be proven. Reduced relation
updates compromising useful expressivity remained a hypothesis, not a measured
cause. The exact uniform depth-scaling mechanism was closed without a scale sweep.

Both candidates helped reference-high-error quintiles but hurt low-error
quintiles. These groups were defined using target-derived reference errors;
their enrichment was post-hoc and could not authorize a deployable router.
Row bootstrap did not estimate training-seed variability or independent-role
generalization. No 500K/full, successor, ensemble or seed confirmation was released.

## RML and resource evidence

Both prospective plans were finalized with canonical 60-observation traces,
role events, measured native allocation cost and retained prediction/checkpoint
hashes. Identical, independently verified reuse views of the same G1 reference
were collapsed in replay enrollment; conflicting identities remained distinct
and subject to the ambiguity gate. Both candidate/reference pairs then appeared
in the actual replay pool with `capability=complete`.

The existing terminal adapter retained `INCONCLUSIVE` for nonpromoted arms;
this report did not rewrite immutable terminal labels or manufacture a negative
policy-backtest truth. Replay-ready here meant retained comparable trajectories,
not an automatically eligible promotion-label backtest or scientific victory.

Wall time was 10,831.975 seconds (3.009 hours); two T4 allocations cost 6.017764
device-hours, including reserved idle time rather than measured GPU busy time.
Mean epoch times were 178.415/178.589 seconds. All six ten-epoch continuation
chunks, last checkpoints and best-model hashes passed acceptance. Official
validation, test-dev and test-challenge roles were untouched. The terminal event
was claimed by A, and the existing heartbeat was paused without creating a chat.

Authority: [independent acceptance](acceptance.json), per-arm
[decay closure](../degree_group_decay_ema999/results/closure.json) and
[pair closure](../degree_pair_depth_scale_ema999/results/closure.json), and the
[physical receipt](../submission_v1.json). No new job was submitted during acceptance.
