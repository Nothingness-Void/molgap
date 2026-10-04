# G1 fixed500K execution prerequisite

Server Kaggle2 qualification authorized on2026-10-04. This is an execution-only
NO_TRAIN experiment with37 explicitly disposable optimizer updates. It consumes
only accepted fixed500K train graphs. No development tensor, selected checkpoint,
official validation or sealed role is read. The immutable100K train-only target
transform and already accepted G1 random initialization are reused unchanged.

Use existing G1 model/optimizer/loss/step, with two independent EMA filter states
observing the same disposable live model. This is not two independent candidate
trainings. Repeat a seeded optimizer step twice and require exact loss/live-state
identity. Measure5warmup+30steps, peak reserved memory, runtime and a train-role
evaluation proxy. Preserve finite outputs and source/fixture hashes.

Report estimated46,860-step /5,998,080-presentation training plus60three-weight
50K evaluation checks. This estimate excludes real dev molecule-size variation,
checkpoint IO and bootstrap. At least15%memory headroom and an estimate at most
4hours are prerequisites, not sufficient training-release certificates. A passed
profile does not open validation, release a long run or promise Replay-Ready.
Long training still requires an executable scale/resume owner, frozen comparison
and reference-acquisition semantics, accepted target/runtime bindings and budget.

Cap2700seconds kernel-entry wall time; count all allocatedT4s, even the idle one.
Save source/runtime/fixture/calibration, actual cost, failures and manifest-bound
small output independently. RML closure is infrastructure-only, never a fabricated
training trace or causal MAE claim. Scalar LR/decay projections are counterfactual
clocks, not observed parameter norms or an instruction to divide decay by five.
