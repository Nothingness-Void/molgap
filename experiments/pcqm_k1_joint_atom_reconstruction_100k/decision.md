# Joint Gap / atom reconstruction: terminal decision

On 2026-09-28 JST, Kaggle2 `kaseichou/molgap-k1-joint-atom-s42:v3` completed
both isolated T4 workers and their frozen-checkpoint audits. Independent saved-
artifact acceptance passed. No local model ran. The exact two recipes were
closed without another seed, coefficient sweep, 500K training or successor.

## Paired endpoints

Both candidates trained on the accepted 100K rows, with the unchanged K1
inference architecture (3,658,817 parameters), seed42, FP32, BS128, 40 epochs,
31,240 optimizer steps and 3,998,720 sample presentations. The 33,582 auxiliary
head parameters were training-only. Both selected epoch39 (zero-based).

| Model / recipe | Original 50K dev MAE (eV) | Frozen500K 50K dev MAE (eV) |
|---|---:|---:|
| Frozen clean K1 reference | 0.14137364 | 0.14125331 |
| A: 1% atom corruption + Gap | 0.14172594 | 0.14444292 |
| B: same corruption + Gap + 0.1 atom CE | 0.13851346 | 0.14187328 |

Original-role selected predictions are the training endpoint authority.
The terminal clean re-inference reproduced them within 1.91e-6 eV per row;
rounding-level differences between the two retained payloads did not change
any decision. Both internal-development roles had already been reused in
discovery; neither was a sealed test. The second column of results did not
involve training on 500K molecules.

- A versus K1: original gain -0.00035230 eV; paired candidate-minus-reference
  95% row interval [-0.00053944, +0.00124980]. On the frozen500K role the gain
  was -0.00318961, with unfavorable interval [+0.00238025, +0.00400290].
- B versus K1: original gain +0.00286018 eV, favorable interval
  [-0.00374760, -0.00195241], but below the prospectively frozen 0.003 gate.
  Frozen500K gain was -0.00061997, interval [-0.00014664, +0.00141132]. That
  interval crossed zero: no portable gain was established, rather than a
  confident claim that B was worse on this role.
- B versus A: gains were +0.00321247 on original dev and +0.00256964 on
  frozen500K dev. Both paired intervals were favorable, respectively
  [-0.00410317, -0.00232234] and [-0.00339605, -0.00174813]. Thus the auxiliary
  term helped conditional on this corruption recipe in both observed roles.

The original material gate and the frozen500K >=0.001 eV / >=50% retention
gate were not passed. Row bootstrap measured molecule sampling, not training-
seed variance; it did not justify changing the frozen thresholds afterward.

## Attribution and limits

The auxiliary objective was active: B's reconstruction CE fell from 0.25659
to 0.08905; matched corruption counts, finite gradients and clean inference
were independently verified. Its original-dev lead over K1 was about 0.00886
at epoch19 but only 0.00286 at epoch39. The late reference caught up; this was
not evidence that a longer continuation would restore a growing advantage.

A's independent-role regression and B's partial recovery support a more
specific interpretation than "auxiliary tasks do not work": reconstruction
counteracted some cost of random categorical corruption. The complete recipe
still failed portable superiority over clean K1. This study did not isolate
auxiliary supervision without corruption, identify a unique chemical failure
mechanism, or measure seed robustness. Ordered 5K-row blocks were descriptive,
not independent scientific splits.

The loss of relative benefit was visible on disjoint audit molecules with the
same 100K-trained checkpoints, before any 500K optimizer update. It therefore
could not be attributed to reduced 500K training exposure in this experiment.
No claim about an actual 500K-trained or full-scale candidate followed.

## Retention and cost

The observed job wall was 7,725.55 seconds (about 2h09m), with 15,451.10
allocated T4-device seconds (4.292 device-hours), not a Kaggle billing reading.
Training throughput was 541.87 / 544.22 graphs/s and peak reserved memory
604 MiB per worker. The two audit intervals totaled 154.96 device seconds;
these were subtracted from training cost and recorded once as a separate
NO_TRAIN action. Parallel audit wall/CPU time was not fabricated. Notebook
bootstrap and idle occupancy was allocated once to A for ledger purposes,
not attributed to its model throughput. Earlier failed/cancelled attempts
retained their own unknown-cost evidence and were not treated as free.

Saved analysis and manifests:
[acceptance](attempts/v3/results/acceptance.json),
[paired and trajectory analysis](attempts/v3/results/analysis.json),
[execution](attempts/v3/results/execution_summary.json).
Raw predictions, checkpoints, recovery archives and logs remained under the
ignored `platforms/_records/kaggle/training/k1_joint_atom_s42_v3/` directory.
The [submission receipt](submission_receipt_v3.json) binds the actual source,
version and dataset mounts. Official validation/test-dev/challenge remained
untouched.

Per-action decisions:
[A](attempts/v3/arms/k1_corrupt_gap/decision.md),
[B](attempts/v3/arms/k1_corrupt_gap_atom_aux/decision.md),
[separate audit](attempts/v3/audit/decision.md).
These were terminal controller decisions, not releases for further compute.
