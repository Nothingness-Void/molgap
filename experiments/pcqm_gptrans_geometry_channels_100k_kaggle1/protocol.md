# GPTrans 100K distance versus distance-angle protocol

## Question and prior evidence

The accepted historical distance-angle geometry GPTrans run and the local
geometry-complementarity screen nominate geometry as useful, but neither has a
strict matched V4 GPTrans reference or a V4 runtime certificate. The frozen
GPTrans-T 100K V4 reference is historical-partial for replay. This screen
therefore tests the narrower unresolved question: **does adding a bond-angle
channel improve a GPTrans already receiving bond distances?** It cannot by
itself establish a geometry gain against pure 2D.

Both arms retain the GPTrans-T 12-layer, 256-node, 32-pair core and add the
same zero-initialized 16-basis distance projection into real-bond pair state.
`distance_angle` additionally enables the zero-initialized 16-basis angle
projection averaged into the wedge-center node. `distance_only` carries the
same dormant angle parameters so the complete initial tensors match exactly;
its angle path stays disabled throughout training and inference. No target or
residual is used to choose a graph-specific route. Invalid generated geometry
is masked by `geometry_valid`, not silently filtered. Training and future
inference must use the same ETKDGv3 plus MMFF94s geometry contract.

## Frozen roles, training and resource gate

Use Kaggle1's accepted private `nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1`
dataset version 4, manifest SHA256
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`.
Its 100,000 train rows have source indices 0–99,999 and its 50,000 disjoint
internal-development rows have 100,000–149,999. Both derive solely from
official training. Official validation, test-dev and challenge remain sealed.

Run the two arms in separate processes on one T4 each, within one Kaggle1
T4x2 kernel. Both use seed 42, the same frozen GPTrans core initial state,
deterministic FP32 with TF32 disabled, global seed-plus-epoch sampler,
physical batch 128, drop-last, 60 epochs, normalized Gap L1, AdamW at 1e-3
with weight decay 0.05, four-epoch warmup then cosine to 1e-6, gradient
clip 1, EMA 0.9999, and best-development EMA selection. Each arm must observe
46,860 optimizer steps and 5,998,080 sample presentations. The only intended
scientific difference is activation of the angle information path.

Freeze source, data, model and initial-state identities before release.
Complete syntax/package, exact real-shard loader and source/mount checks
locally. Both remote workers must independently pass repeatable finite
physical-batch optimizer steps, a runtime certificate, at least 15% memory
reserve and estimated training duration no more than six hours per arm before
either worker trains. A failed preflight blocks both arms. Checkpoints and
selected models must be durable and independently retrievable.

## Paired scientific decision

On the exact aligned 50,000 development rows, compute per-row absolute-error
differences `distance_angle - distance_only`. Use 10,000 paired row-bootstrap
replicates with seed 42 for a two-sided 95% interval of the mean difference.
The angle channel passes this bounded shortlist gate only if its MAE improves
by at least 0.003 eV **and** the interval upper bound is below zero. Row
bootstrap measures row uncertainty, not training-seed variation. A pass is
only a nomination for a separately frozen scale decision. A failure closes
this angle increment under the current 100K contract. No automatic second
seed, 500K/full run, official evaluation or production change follows.

## Acceptance

Independently accept each arm's exact source/data/runtime identity, 60-epoch
trace and exposure, selected model, aligned finite development predictions,
resumable checkpoint, measured native T4 cost, and exact train/development
role use. Bind the candidate trace and strict V5 comparison to the accepted
same-run reference evidence. Use the existing GPTrans acceptance,
`experiment_cli terminal`, and RML commands. Two distinct replay-pool entries
must show `capability: complete` and empty `exclusion_reasons` before calling
the pair dual replay-ready. A kernel queue state is not acceptance.
