# Terminal attribution —2026-10-15 JST

Authority: [decision](decision.md), [analysis](analysis.json), frozen recipes
and retained trace/selected/prediction/resume/native/cost records. No inference.

K1 Clean development is best0.109049469 at pass32, then0.110085443 at60. Live
train MAE, converted with frozen target std, falls0.095083→0.076397eV over that
tail.28 additional passes did not improve development: this exact run did not
stop before its planned exposure. This is not a universal K1 capacity ceiling.
Live train uses dropout and weights changing during updates; development uses
fixed Clean EMA. Their gap is not a matched-state/cohort generalization metric.
Overfitting, EMA lag, calibration-prefix effects, single-forward regularization
loss and cross-job recipe/runtime effects remain insufficient_evidence as
individual causes. No matched raw/live development curves were retained.

Clean integrity passes:18 BN modules each have128 calibration batches; other
selected tensors equal same-step raw EMA; state/revision/hash bindings agree.
Native repeat/resume/calibration proofs match. No observed corrupt artifact or
stale Clean state explains the score. This does not prove scientifically optimal
calibration distribution or EMA decay.

GPTrans development is best0.101951368 at45 and0.102401979 at60, while live
train falls0.089483→0.073029eV. This tail shows no late development recovery.
Its endpoint is approximately0.064meV worse than historical EMA0.101887690,
context only: no strict reference was frozen. Local-bond causal benefit/harm
at500K is insufficient_evidence.

K1 process wall10.401814h versus GPTrans11.470336h,9.32% less wall at equal
gradient exposure. That includes admission, evaluation/calibration/checkpoints;
it is not isolated kernel speed. Individual canonical pass timings are absent.

K1 is7.098100meV worse but wins45.472% of rows. Residual correlation0.847077;
fixed blend gains1.544446meV. Complementarity survives; this weaker K1 limits
the blend. Mean signed errors+0.006433/+0.005798eV. No fitted correction/router/
MoA/tuned blend or protected-role consumption.

Do not repeat this pair or launch full training from these endpoints. Preserve
the stronger retained blend. If separately authorized, the cheapest K1
discriminator is fixed same-checkpoint Live/Raw/Clean evaluation using retained
same-step selected/terminal states on an authorized existing role. It can
separate inference-state effects without another60-pass run; missing training
controls and seed variance remain unresolved. RML retains INCONCLUSIVE.
