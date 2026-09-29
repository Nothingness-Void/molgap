# GPTrans-T input initialization 100K launch decision

Decision date: 2026-09-29 (Asia/Tokyo).

The user requested the additional experiment needed to select low-cost
GPTrans modules and prioritized small training and inference time increments.
The accepted V4 reference and RML family decisions identify a previously
untested, zero-parameter intervention: change only the initial scale of its
input embedding tables. PairNorm and Noisy-related 100K gains did not transfer
at 500K; no repeat or combination of those modules is released here.

Select one desktop-owned 100K/50K candidate under `protocol.md` and
`training_contract.json`, reusing the accepted GPTrans-T V4 reference rather
than retraining a control. This choice minimizes new GPU training, but the
P100 historical reference and any different candidate runtime make the
endpoint comparison a screen, not a same-allocation causal estimate.

At this decision, the candidate model-state hash, committed executable
source, platform mirror/run identity, prospective trajectory, and remote GPU
certificate were not yet bound. The user's local-analysis scope does not
submit or release GPU work in this task. If remote execution is later directed,
one candidate GPU preflight and, conditional on its source, data, role,
repeatability, memory, latency and native-cost gates, one 60-epoch 100K
training attempt are the only contemplated actions. Failed or missing gates
do not silently authorize training. No protected role, second seed, 500K,
full-scale training, or automatic promotion is released. A terminal result
requires acceptance, attribution, actual native cost, and RML closure.
