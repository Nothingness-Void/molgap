# Evidence review

Use ../pcqm_k1_norm_exposure_100k/terminal_decision.md and raw160-row curves.
OriginalLR LN is5.771meV worse than BN. LowerLR+warmup improves same-job
LN by2.973meV; same-role scalar bias removal does not explain this benefit.
LN fits lower online train loss while worse on development under originalLR;
simple insufficient train fitting is not established. This does not prove
overfitting, nor show WD is causal. Stronger decoupled regularization is the
new bounded learning-constraint hypothesis; no new structural module is invented.
ModerateWD1e-4 and strongerWD1e-3 are both untested under this exact warmup
contract; frozenWD1e-5 baseline is reused as context, not retrained.
Do not reopen negative consistency/distillation/slot96 routes. Reuse exact
archived18local PostLN surgery69c91dbc with generic composed addon dispatch;
family trainer,compiler,source packaging,submitter and acceptance stay shared.
