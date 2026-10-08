# Connected/local relation-flow dual — 2026-10-08

MOLGAP-COMMON-V5-FINAL, server-owned. User authority: "我希望提交一个双100K的
试验 来寻找新方向". Release exactly two new seed42 candidates in one T4x2
notebook. Reuse the accepted local-bond100K reference; do not retrain a baseline.
No500K optimization, extra seeds, full training, geometry, teacher, pretraining,
Router, protected evaluation or automatic successor is authorized.

## Evidence and two distinct hypothesis cards

The [local-stream screen](../pcqm_gptrans_capacity_relations_100k/gpu/results/interpretation.md)
was directionally positive against G1 at100K. Its [same-update500K bridge](../pcqm_gptrans_local_transfer_500k/gpu/results/interpretation.md)
retained a material benefit. These observations justify testing information flow
on that parent, not further node/FFN width or final readout replacements.
The [frozen audit](../pcqm_gptrans_bottleneck_audit/decision_terminal.md) found all
local branches active, but the original real-pair transition's final branch
disconnected from the virtual readout. The [amplitude cap](../pcqm_gptrans_local_control_100k/gpu/results/interpretation.md)
worsened its endpoint. No cap, width, decay or readout grid is reopened.

### A: connected valid-pair transition

- Deficiency: GPA evolves pairs through head projection; an independent pair
  nonlinearity was not fully tested at the last block because its virtual row
  was excluded. This does not prove pair rank collapse.
- Change: before local/GPA blocks3/6/9/12, pair + zero-return
  `Linear64to32(GELU(Linear32to64(LayerNorm32(pair))))` on **all valid pairs,
  including virtual rows/columns and[0,0]**. Padding is excluded.
- Parent: preserve all12 local branches, core tensors and virtual readout.
  Total5,888,225 parameters,17,024 above the5,871,201 parent.
- Cheapest falsifier: optimizer-inclusive remote preflight must find nonzero,
  finite return-weight gradients at all four insertions before full training.
- Alternative: GPA already supplies enough relation freedom; dense pair FFNs
  may increase cost or overfit. Connectivity is necessary, not proof of benefit.
- Decision changed: whether *connected* independent pair processing warrants
  retention. This is a new scope/parent comparison, not a retry or retroactive
  repair of the [closed original transition](../pcqm_gptrans_pair_transition_100k/gpu/results/interpretation.md).

### B: sparse true-bond return

- Deficiency: the accepted64-channel bond message updates nodes but does not
  itself write chemical content back into the32-channel pair state. GPA already
  updates pairs, so the additional channel could be redundant.
- Change: in each of12 local blocks, use its existing bond-message hidden vector
  to produce a zero-initialized `Linear64to32` pair residual only on the accepted
  directed true bonds. Recompute the parent's local message with the updated
  pairs in the **same block**, then call its unchanged GPA core.
- No new graph representation, edge cache, angular/triplet feature, persistent
  extra64-channel memory or additional all-pair FFN. The second local-message
  evaluation is an intentional compute cost, not hidden in parameter count.
- Total5,896,161 parameters,24,960 above the same parent.
- Cheapest falsifier: all12 return gradients must be connected in native remote
  preflight. Without same-block consumption, the final real-bond-only update
  could repeat the earlier disconnection; that implementation is forbidden.
- Alternative: existing GPA pair updates already suffice; recomputation may
  worsen throughput without improving generalization.
- Decision changed: whether chemical local messages need a direct bidirectional
  node/bond/pair route, rather than a second independent dense pair MLP.

## Literature mechanism review and limits

[GPTrans, sections3.2–3.3](https://www.ijcai.org/proceedings/2023/0396.pdf)
explicitly couples node-to-node, node-to-edge and edge-to-node propagation and
avoids a separate dense edge FFN for efficiency. A tests a bounded deviation
from that choice; its cost warning is contrary evidence, not ignored evidence.
[GPS++, section4.1](https://arxiv.org/pdf/2302.02947) updates local edge states
from endpoint/edge/graph states and uses those messages in node updates. B is a
small pure2D local-return hypothesis inspired by that ordering, **not** GPS++
reproduction, its pretraining recipe, its3D input or its published score.
Neither paper demonstrates these exact small addons on this100K contract.

## Immutable contract and reference

Retain the accepted [local reference qualification](../pcqm_gptrans_local_control_100k/reference/reference_bundle.json)
or enroll the same terminal candidate additively using the existing reference
owner. Release verifies real pointers/hashes; a scalar cannot qualify it.
Both candidates compare to the local parent, not a weaker G1 scalar.

Match fixed cross-platform100K manifest, train[0,100000), development[100000,150000),
OGB atom9/bond3/SPDcap20 consumed inputs, seed42, FP32/noTF32, physicalBS128,
drop-last32, accumulation1, deterministic algorithms, portable train100K target
transform, normalized Gap-L1, AdamW single group lr1e-3/wd0.05/foreach=false/
fused=false, clip1, dropout/drop-path0.1,60 full100K passes, warmup4/cosine60/
min1e-6 and EMA0.999 each step. Exactly46,860 updates/5,998,080 presentations.
Only architecture/config identity differs for each independent comparison.

Initialize from accepted **scratch initial** local tensors, never trained
weights. Preserve every parent tensor byte; added returns start at zero and
added hidden tensors use a forked seed42. No new forward RNG consumption.
Tensor-only CPU packaging is allowed; actual model math remains remote.

## Interpretation, cost and stop rules

Keep the matching benchmark's0.003eV material nomination gate and positive
paired-row bootstrap lower bound. It is not a universalV5 threshold or measured
training variation. Independently selected EMA endpoints, final-ten curves,
live/EMA gaps, fitting/generalization, returned-update RMS and per-layer return
gradient/weight norms are retained. Scientific decisions are per arm; one
failure cannot promote or erase the other. No combinations are selected here.

Estimate4–6 notebook wall hours /8–12 allocatedT4-hours; hard maximum7wall/
14allocatedT4-hours. Account2 quota and exact identity must be reconciled before
POST. Native optimizer-inclusive preflight requires estimated training<=5hours
per worker and at least15% VRAM headroom; reserve evaluation, hashing and output
publication overhead. If either preflight fails, retain its failure and release
no silently changed recipe. Two independent processes each isolate one T4,
model, RNG, optimizer, sampler, EMA and output directory. No P100 request.

Record prospective identities/reference/roles/trace/budget before submission.
Native outputs retain60 canonical observations, aligned50K prediction/target/
source-index tensors, exact selected weights, full resumable last state and
atomic ten-epoch chunks. Retain source/runtime/initial/transform and all artifact
hashes, actual wall/allocation/queue/CPU cost or explicit unavailable semantics.
Reuse native saved-output acceptance, terminal/reference transaction, RML
validate/rebuild/frozen checks and **actual per-arm complete Replay admission**.
Prelaunch eligibility is not a completed Replay result. Missing evidence stays
missing; no forced promotion, protected access or blanket historical repair.

A separate frozen-weight500K cohort audit may be planned after acceptance, but
this protocol does not release it or500K training.100K victory alone is not a
scale-transfer claim. Reuse existing Luna B after exact returned job/version
reconciliation: healthy silent, one durable terminal/fault event to existing A.

## Reuse scope

Only `gptrans_local_relation` implements new math. Native trainer dispatch and
diagnostic hooks, reference-qualified release, source/check-release, Kaggle
bootstrap/all-arm preflight, saved-artifact acceptance and RML closure are reused.
Do not copy a trainer, generic graph recipe, platform submitter or monitor loop.
