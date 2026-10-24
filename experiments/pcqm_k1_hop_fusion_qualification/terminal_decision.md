# Terminal saved-fusion decision — 2026-10-25 JST

Close this diagnostic **NO_TRAIN** after one successful CPU arithmetic run.
Four retained payloads passed original file hashes, exact50000 ascending IDs,
identical finite float32 Gap target bytes and canonical row/target hashes.
No checkpoint/model execution, gradients, optimizer or remote download occurred.

| Fixed predictor | Development Gap MAE eV |
|---|---:|
| Retained Mean2 + Consistency2, equal |0.134665992531|
| Hop mixed + ordered, equal |0.136510747544|
| Four experts, equal |0.131565311916|

The hop pair alone is1.844755meV worse than the retained equal pair; paired-row
95% interval[1.102677,2.582134]meV. Four equal experts improve the retained pair
by3.100681meV; candidate-minus-reference interval[-3.485485,-2.711183]meV,
2000 draws/seed42. Their equal-blend signed-residual correlation is0.897514.

The four-expert point difference meets this diagnostic's3meV prioritization
floor and its upper interval is below0. It does **not** establish a confidence
bound of at least3meV: the shared helper's stricter upper<=-3meV predicate is
false. Neither predicate is a new model promotion, seed-variance or scale gate.
All model endpoints had previously been selected using this development role.

## Disposition and next decision

Preserve four-way fixed output fusion as a numerical ensemble candidate.
Do not adopt the hop pair as a replacement for the existing stronger pair.
This supports prediction complementarity, not additive gains from inserting
MsgNorm, shell recurrence, consistency and pretraining into a single model.

Do not submit a repeat100K shell/MsgNorm pair or an all-module500K package from
these observations. The user-approved next100K integration step was conditional
on a qualified, decision-relevant combination. That condition is not established:

- The retained pair has unresolved strict native-cost/continuation exclusions;
  this arithmetic cannot repair its training evidence.
- Four expert serving cost and a same-size alternative ensemble comparison are
  unmeasured; no compute-efficiency claim or matched model-count gain follows.
- The original accepted K1 payload and historical K1/PairToken prediction matrix
  are missing locally. Historical0.1340852096 is contextual only; four-way's
  arithmetic point difference versus it is2.519898meV, without a paired interval.
- One repeatedly used development cohort does not establish independent transfer
  or prove the same complementarity at500K.

Before authorizing expensive scale work, require an explicitly frozen expert
set, an available qualified reference and a budgeted model-count/cost comparison.
These are the missing discriminators; baseline retraining or a new arbitrary
module is not released to fill them. Original negative training decisions remain
unchanged. No production adoption,500K/full run or protected role is released.

## Cost and custody

Worker wall3.024514s; process CPU3.75s. These native timers exclude imports,
planning, Git, historical training and the independent metadata check. GPU/queue
are not applicable. Original prediction inputs total3213332bytes and were reused
in place on D:. Generated fixed predictions and compact analysis total1808714
bytes before terminal documentation. No training replay-ready claim is made;
both original hop training entries keep their independent strict qualification.
