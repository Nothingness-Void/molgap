# Local relations and equal-update500K EMA: terminal interpretation

On 2026-10-05, the two version1 arms were independently accepted from retained
outputs. Source/receipt/config bindings, atomic checkpoint chunks, full canonical
traces, finite aligned saved predictions, role events and native allocation cost
were verified. Exact values, milestones and Replay proof are retained in
[interpretation.json](interpretation.json). Mechanical authorities are
[local acceptance](local_acceptance.json) and [scale acceptance](scale_acceptance.json).
No local model loading, training, inference or protected-role access occurred.

## Real-bond local stream: positive but below the frozen gate

| Endpoint | Parameters | Selected epoch, zero-based | Saved-prediction Gap MAE, eV |
|---|---:|---:|---:|
| Immutable G1+EMA999 reference | 5,246,817 | Frozen reference selection | 0.1442326291 |
| Real-bond local stream | 5,871,201 | 41 | 0.1423582275 |

The selected improvement was0.0018744017 eV. Candidate-minus-reference paired-row
95% interval was[-0.0028262450,-0.0009326870] eV;50.532% of rows improved.
This is a directional single-seed result, not seed stability. It did not pass
the prospectively frozen0.003 eV material gate. The immutable terminal
INCONCLUSIVE/no-promotion label was retained; no threshold was lowered afterward.

The addon used roughly11.9% more parameters. It improved early fitting and
development together: at epoch19, training MAE was0.142798 versus0.156984 for
G1, and EMA development was0.152672 versus0.159538. At epoch41 the candidate
selected its best model. Continued training reduced train MAE from0.094091 to
0.074937, while EMA development worsened from0.142358 to0.142773. The reference
continued catching up. This supports useful local information flow but a limited
generalization payoff under the frozen budget; it does not show that more width
or more steps would restore a material gain.

The exact local comparison was STRICT_CAUSAL with no blockers. Candidate and
the immutable G1 reference were actually found in the rebuilt Replay pool with
capability=complete, no exclusions and identical comparability keys. Replay
readiness did not imply scientific promotion. No500K local-addon training was
performed: the companion scale arm used unchanged G1, not this addon.

## One live500K encoder, two EMA views

| Selected view | EMA decay | Best rung, zero-based | Saved-prediction Gap MAE, eV |
|---|---:|---:|---:|
| Slow filter | 0.9999 | 59 | 0.1247831724 |
| Fast filter | 0.999 | 59 | 0.1152574413 |

Fast-minus-slow selected MAE was-0.0095257311 eV, with paired-row95% interval
[-0.0099073039,-0.0091514034];59.31% of rows improved. Both predictions used
the same fixed internal-development rows500000:550000, with equal targets.
The optimizer/sample/LR/live-train/live-development/checkpoint fields were equal
at all60 rungs. There was one5,246,817-parameter live encoder, not two independent
training streams. EMA correction added no inference parameters.

The endpoint completed46,860 updates and5,998,080 presentations: approximately
12 fixed500K passes, not60passes. Fast EMA ended only0.0000266 eV above live
development. During the last ten rungs, slow EMA improved0.0168186 eV while
fast EMA improved0.0008594 eV. Slow averaging substantially lagged this live
trajectory. Initial-state retention versus smoothing lag was not separately
intervened on; neither mechanism's individual contribution was measured.

The earlier [100K EMA intervention](../../../pcqm_gptrans_input_ema_100k/gpu/results/decision.md)
and this genuine500K optimization observation agree directionally. This is
evidence that correcting weight-selection dynamics can survive enlargement.
It is not proof that an architecture gain survives, that500K training converged,
or that G1 beats K1/EdgeState at full scale. Absolute100K and500K MAEs were not
subtracted: their development molecules differ. No official-validation ranking
was inferred from this internal role.

The frozen purpose was transfer_study with no accepted same-contract500K causal
reference. Acceptance remained PAIRED_ENDPOINT, strict_ready=false and
replay_ready=false. One prospective physical-run trajectory was RML-finalized,
retaining both views but only one primary canonical trace. It was honestly
excluded from the causal Replay pool. Its valid INCONCLUSIVE trajectory outcome
is separate from its PAIRED_ENDPOINT comparison qualification.

## Four-study synthesis and disposition

The [node-width/FFN capacity falsifier](../../../pcqm_gptrans_capacity_nodes_100k/gpu/results/interpretation.md)
did not support promotion either. Across the three100K architecture packages,
only the real-bond interior stream was directionally positive, and it remained
below gate. Parameter count alone was not a supported explanation for the
remaining error. The stronger result was EMA correction, supported here after
actual500K optimization without enlarging the predictor.

The two notebooks consumed13.682878275 allocated T4 hours total. This notebook
consumed13,530.695089 wall seconds and7.517052827 allocated T4 hours. Each physical
arm received half the complete notebook allocation in native RML accounting;
both EMA views belonged to the same physical arm and were not charged twice.
No utilization was inferred from allocation or wall time.

The bounded campaign closed without seed expansion, continuation, protected
evaluation, desktop custody transfer or an automatic successor. Local geometry,
Router and ensembles were not authorized by target-derived residual quintiles.
Any separately reopened local-addon transfer test needs a frozen role/compute
decision; it cannot borrow the unchanged G1 scale arm as its own qualification.

## Acceptance repairs and verification

The accepting Windows host differed from the remote Linux cosine result by
one ULP at rung18. The adapter accepts at most two ULPs when recomputing the
frozen LR; steps/presentations and equality between observed EMA views remain
exact. Native source/config/trace bytes were not rewritten. A second adapter
repair replaced an unsupported CONTEXT_ONLY trajectory outcome with valid
INCONCLUSIVE, without changing the noncausal comparison or scientific contract.

Three focused synthetic acceptance tests passed. Repository RML validate and
check --frozen passed after terminal publication. Actual local/reference Replay
admission and500K exclusion were checked independently. No cross-machine
portable audit, local model execution or remote resubmission was performed.
