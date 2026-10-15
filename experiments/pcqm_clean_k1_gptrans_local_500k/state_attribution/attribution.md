# Failure-mode attribution

Authority:[terminal decision](terminal_decision.md), [results](results/result.json),
[protocol](protocol.md), hash-bound source/checkpoints/data and
[recipe audit](recipe_comparison.md). Frozen-state interventions, not training.

Supported: the current selected EMA weights are less accurately evaluated using
their retained prefix-Clean buffers than their Raw buffers. Changing only that
saved BN state worsens consumed50K MAE by3.653meV. Uniform calibration nearly
recovers it. Independent same-CPU Live prefix/uniform interventions establish
cohort sensitivity; they do not prove the exact chemical reason for it. No
stale/broken artifact is observed. CPU replay agrees with retained native Clean;
other views remain local FP32 diagnostic outputs, not new native qualification.

EMA-weight and buffer effects differ: under the identical uniform calibration,
EMA beats Live by5.117meV. A generic claim that EMA or single-forward training
must be harmful is unsupported. From selected to final, raw EMA development
worsens0.620meV despite lower clean sampled train error, while Live development
improves4.067meV. Hence blanket late-overfitting attribution and a new automatic
early stop are not justified by these two endpoints.

Unidentified: pretraining, one/two forwards, consistency, schedule, optimizer/
runtime, and selection effects in the residual historical difference. The old
training used a pretrained backbone and a different loss/selected state; no
matched training control exists here. Low pooled rank does not identify a slot
capacity ceiling. Selection-used development and row CIs are not independent
validation or seed uncertainty. Preserve original terminal conclusions.
