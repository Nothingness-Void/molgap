# K1 dropout consistency bounded contract — 2026-10-04

User authority: organize completed FLAG evidence, submit one justified new
overnight question on Kaggle3, then shut down desktop after verified submission.
Shutdown does not transfer ownership or create a monitor. This question has two
new equal-pass objectives, not a clean baseline retraining or FLAG parameter retry.

Preserve original pure2D K1: atom192,bond64,slot64,active slot1,nine blocks,
mixers3/6/9,RWSE16,mean pooling,3658817 parameters and clean inference.
Same batch and weights produce two independent dropout predictions p1,p2.
dropout_mean2: mean of the two normalized Gap L1 losses.
dropout_consistency2: that loss +0.1*mean((p1-p2)^2), in normalized Gap space.
Both predictions participate in differentiation; one clip1.0 and AdamW update.
No detach,perturbation,new weights,geometry,pretraining or target replacement.

Frozen cache nvoid912/pcqm4mv2-ogb-fixed-100k-v1, manifest
1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d.
Only train[0,100000) and already consumed development[100000,150000).
No official-valid,test-dev,test-challenge or server role is released.
Seed42,frozen original initial state,FP32/noTF32/deterministic,BS128/drop-last,
Python epoch order,AdamW lr4e-4/wd1e-5,cosine40/eta1e-6,40epochs,noEMA,
best clean development live checkpoint. Each arm completes31240updates,
3998720 optimizer-batch presentations and7997440 gradient loss-row evaluations.
The extra consistency loss uses the same two predictions; actual time is measured.

Before formal training: local synthetic objective/gradient/resume checks,
syntax/package/immutable input acceptance; prospective publication; then real
train-only BS128 remote qualification for both assigned T4s. Require finite
nonzero dropout disagreement and model/disagreement gradients,repeatability,
RNG/model/AdamW/cosine next-step resume and clean selected-state roundtrip.
No local train256 model diagnostic is claimed. The all-arm barrier prohibits
either formal arm starting if any qualifier fails. Remote calibration step ratio
relative to clean K1 must be <=3.0. Qualification failure is NO_TRAIN/resource
failure, not a scientific endpoint. Expected each arm<4 assigned T4h,total<=8T4h;
these are estimates/budget ceilings,not measurements. Kaggle wall limit12h.
Record T4,CPU,queue,bootstrap,idle/allocation scopes independently; unknown remains
unknown. No duration truncation,automatic successor or calibrated early stop.

Primary mechanism gate: mean2-reference minus consistency-candidate>=3meV and
positive lower95% paired row-bootstrap bound (1000 draws,seed42) on aligned50K.
Historical clean K1 is separately checked against the same3meV/sign nomination
gate; it is not the primary equal-pass mechanism comparator. Historical runtime
differences exclude strict causal/READY claims. A historical gain alone cannot
establish consistency contribution. Single-seed training stochasticity is unknown.
Complete40epochs before scientific judgment. Post-hoc slices remain descriptive.

Pass: review strict runtime/reference,cost and transfer evidence separately;
no automatic500K/full,official evaluation,promotion or submission. Fail: terminal
attribution,complete-history archive and accepted desktop canonical discovery.
Mean2 outcome and consistency contribution are separate. No seed/schedule retry.

Freeze source/config/input/initial hashes; retain atomic model/optimizer/scheduler/
RNG/cursor checkpoints,selected weights,aligned predictions,full trace,runtime
certificates,observed native costs and exact kernel/version in retrievable manifests.
Same-run replay binding records the paired ownership only; replay eligibility
still requires the existing strict comparison-readiness validators.
