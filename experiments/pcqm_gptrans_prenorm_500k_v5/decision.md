# GPTrans Pair PreNorm 500K trajectory stop decision

Decision date: 2026-09-19

## Outcome

The matched 500K run is closed without another continuation. The original
allocations reached the 16-hour wall with valid atomic checkpoints, and the
certified continuations later failed before adding an epoch because restored
EMA tensors remained on CPU while the models were on the accelerator. The
resume defect is infrastructure-only and does not invalidate the completed
trajectory.

The existing curve is nevertheless sufficient for the authorized cost stop.
Across the 17 common epochs from 20 through 36, Pair PreNorm was worse than the
reference at every epoch. Its mean candidate-minus-reference development MAE
was `+0.0012765996 eV`; over epochs 30 through 36 the mean was
`+0.0013958897 eV`. At common epoch 36 the gap was `+0.0014580414 eV` in the
wrong direction.

Late convergence also did not favor the candidate. Linear development-MAE
slopes over epochs 30 through 36 were `-0.0003709798 eV/epoch` for the
reference and `-0.0003269733 eV/epoch` for Pair PreNorm. To satisfy the frozen
`0.003 eV` gain gate relative to the epoch-36 reference, the candidate would
need to reverse `0.0044580414 eV` from its matched epoch-36 position while the
cosine learning rate continues to decay. Nothing in the completed late curve
supports that reversal.

## V5 decision

- execution: `PARTIAL_CHECKPOINTED`;
- artifacts: `PARTIAL_ACCEPTED` through reference epoch 36 and candidate epoch
  38;
- strict comparison: `NOT_AVAILABLE` because terminal aligned prediction
  bundles were not produced;
- science: `INCONCLUSIVE` rather than a completed-run negative claim;
- transfer: `NOT_READY`;
- budget: `STOP_FOR_COST`;
- desktop handoff: `NONE`.

The prior 100K gain of `0.0031248707 eV` therefore remains a narrow
scale-specific shortlist observation, not a transferred 500K result. Finishing
would require about `9.7` additional reference hours and `8.5` additional
candidate hours, with no supported path to the promotion gate. Do not repair
or resubmit this continuation, add seeds, read protected roles, or release a
full-scale handoff.

Compact machine-readable curve evidence is in
`results/trajectory_stop_summary.json`. The preserved remote checkpoints and
logs remain the authority for execution provenance.
