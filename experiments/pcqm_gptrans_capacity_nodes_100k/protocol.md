# Corrected GPTrans: node capacity falsification

Frozen on 2026-10-05 under MOLGAP-COMMON-V5-FINAL, server-owned.
User authorization: Kaggle2, two concurrent T4x2 notebooks, four physical
experiments TOTAL (including one 500K shared-live EMA study). No successor,
new seed, full training, official validation or test role is authorized.

## Evidence and question

Retain the [G1 + EMA999 reference](../pcqm_gptrans_input_ema_100k/gpu/results/decision.md)
for research only. Degree conditioning and corrected averaging improved its
selected predictions; readout, decay and small path interventions did not.
The [EMA portability audit](../pcqm_gptrans_ema_portability/attempt_v4/decision.md)
does not establish 500K optimization. EdgeState remains the full delivery
reference. This final capacity experiment does not promote GPTrans by default.

Two isolated arms distinguish global node width from nonlinear FFN capacity:

| Arm | Single architecture question | Initialization |
|---|---|---|
| node352 | Is the 256-channel node stream a representation bottleneck? | Seed42 same constructor/degree scaling policy, newly frozen full tensors |
| ffn2 | Is the unit-ratio node FFN too narrow, without widening GPA? | Accepted G1 shared tensors; appended hidden coordinates with zero output columns |

The wider arm necessarily changes tensor shapes/RNG consumption. It does NOT
claim identical initial predictions. FFN expansion preserves shared tensors,
but enlarged dropout changes the training RNG trajectory. These are declared
architecture-package effects, not pure parameter-count attribution.

## Contract and comparison

Use the accepted fixed100K train and internal50K development asset; direct Gap,
seed42, physical BS128, FP32/no TF32, normalized L1 using the portable accepted
train100K transform, AdamW lr1e-3/wd0.05/foreach=false, clip1, 60 epochs,
warmup4/cosine/min1e-6, EMA0.999 after each optimizer step. Exactly 46,860 updates
and 5,998,080 presentations per arm. Same frozen reference, no baseline rerun.
Only architecture identity changes. Verify actual supplied reference evidence
at release. Material selection gate is 0.003 eV and positive paired bootstrap
lower bound; neither measures seed variation or authorizes automatic expansion.
Record all predictions, row/target hashes, live/EMA trajectories, checkpoints,
runtime, source, applicable roles, native costs and terminal decisions. Require
actual Replay-pair admission at terminal closure, not a prelaunch claim.

Parameter ceiling is 10,493,634 (twice 5,246,817). No geometry, teacher,
pretraining, path sidecar, prediction fusion, depth or optimizer intervention.
Related negative routes stay closed. Extra capacity may overfit or converge
too slowly; the experiment can fail its scientific OR cost gate.

## Budget and execution

One notebook, two independent visible T4 workers. Maximum 7 hours wall / 14
allocated T4 device-hours. The companion notebook has the same maximum, so
the campaign ceiling is 28 T4 device-hours including idle/failed allocation.
Estimated wall 4.5 hours here, not a guarantee. Reject a worker if optimizer
calibration predicts over 6 hours or reserved memory headroom is below15%.
Atomic full optimizer/RNG/EMA checkpoints each epoch and independent ten-epoch
chunks. Infrastructure errors retain evidence; reconcile submission uncertainty
before any retry. No automatic new architecture or modified scientific contract.

## Literature rationale and stopping rule

[GPTrans](https://www.ijcai.org/proceedings/2023/0396.pdf) separates node and pair
capacity and uses a unit-ratio node FFN; its official budget is not our screen.
This experiment tests capacity allocation, not a faithful reproduction or a
guaranteed improvement. Negative results close these exact configurations;
positive100K outcomes only warrant controller interpretation and a separately
qualified transfer decision.
