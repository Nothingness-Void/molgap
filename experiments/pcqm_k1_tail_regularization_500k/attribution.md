# Tail regularization attribution

## Conclusions supported by this run

1. The starting weights were loaded correctly. Both arms reproduced every retained
   full50000 development prediction exactly before any update. A wrong checkpoint,
   transform or initial inference implementation is not supported as the cause.
2. Both20-pass tail recipes fail. Every observed LIVE and shadow-EMA development
   point is worse than the initial LIVE endpoint. Best LIVE occurs at pass15 in
   both arms; final pass20 loses another0.618/0.850meV versus its selected point.
   More late exposure did not deliver a useful endpoint in this frozen budget.
3. Training was active: normalized stochastic objective falls from0.0634515 to
   0.0575530 and0.0634348 to0.0575625. This is dropout/training-mode loss;
   it is not clean fixed-cohort training MAE and cannot establish an overfit gap.
4. Stronger weight decay has a small conditional selected-state benefit,
   0.212meV with positive paired-row CI, but falls short of1meV and does not
   recover the initial predictor. The final wd1e2 endpoint is0.0202meV worse
   than wd1e3. No robust final-endpoint superiority or seed-level claim follows.
5. Raw shadow EMA does not rescue the observed curve. Its best values are
   0.1057856604/0.1057664827eV, both worse than the0.1049040854eV start.
   No alternate EMA endpoint is selected after seeing the result.

## Attribution limits

Only weight decay differs within this new pair. Relative to the old parent,
both arms also restart optimizer state, use a new cosine/LR horizon and remove
the old two-forward consistency objective. Thus the initial regression and
failure versus the parent belong to this combined tail recipe; they do not
prove a unique failure of weight decay, single-forward training or K1 capacity.

Both first updated endpoints regress by2.815/3.484meV and later recover part
of that regression. This is consistent with a restart/state disturbance but
does not identify which changed mechanism causes it. Endpoint and curve facts
do not establish underfitting, generic overfitting, BN harm or exposure shortage.

The exact uniform16384TRAIN start/selected/final and TRAIN-only BN calibration
follow-up could distinguish weight and buffer effects, but its full500K graph
input was retired. Three retained TRAIN shards are not the same cohort.
That discriminator remains insufficient_evidence; no substitute is presented.

## Decision consequence

Close these two tail recipes and lower priority for additional exposure or blind
weight-decay escalation from this parent. Do not scale them to full data or
promote them. Preserve the parent predictor and the negative paired evidence.
This bounded result does not reject every K1 regularizer or architecture.

Owners: protocol.md; decision.md; analysis/analysis_summary.json;
analysis/initial_comparison.json; exact source/receipt and acceptance records.
