# Pure-2D triplet communication dual100K — 2026-10-09

Server-owned Track C, MOLGAP-COMMON-V5-FINAL. User authority: "还有继续研究方向吗
继续提交", following the requested dual100K screen. Release one Kaggle2 T4x2
notebook with exactly two independent seed42 arms. No baseline retraining,
500K/full optimization, extra seeds, teacher, geometry, pretraining, protected
evaluation, desktop custody or automatic successor is released.

## Evidence, literature and bounded question

The accepted [local-stream500K bridge](../pcqm_gptrans_local_transfer_500k/gpu/results/interpretation.md)
retained an equal-update architecture benefit, without convergence/full-ranking
qualification. The [connected pair study](../pcqm_gptrans_local_relation_100k/gpu/results/interpretation.md)
found a small below-gate gain from independent pair nonlinearity, while a
same-block chemical bond return worsened development. Neither measured a need
for pair-to-pair communication; that is the hypothesis being tested, not a fact.
The [frozen audit](../pcqm_gptrans_bottleneck_audit/decision_terminal.md) requires
a connected final virtual-pair route. Width/FFN, readout, amplitude cap and
duplicate local chemical feedback routes remain closed.

[TGT sections3.1 and4.4](https://arxiv.org/html/2402.04538v2) introduce inward
and outward pair-to-pair interaction with a third-pair bias/gate. Aggregation
removes query/key matching; attention retains it. Their ablation distinguishes
capacity from compute cost, but its distance-prediction setting and geometric
training stages differ from this experiment. We borrow only the communication
equations. No published score, geometry accuracy, practical subcubic complexity
or expected Gap improvement is inferred for these small GPTrans addons.

## Two hypothesis cards

Both preserve the accepted uncapped real-bond local parent, not the newly
below-gate connected-pair candidate. Insert a prenormalized zero-return residual
before existing local/GPA blocks3/6/9/12, including every valid virtual pair and
[0,0]. Two heads of8channels operate on32channel pairs; concatenate inward and
outward returns and project32to32. Padding contributes neither values nor
weights. No additional dropout or forward RNG draws. All original tensors and
the node/virtual readout remain unchanged.

### A — third-pair-gated aggregation

- Deficiency/question: pairs exchange primarily through node/GPA channels;
  can direct relation aggregation add information without query-specific scores?
- Change: inward return to(i,j) aggregates(j,k), with softmax overk of bias(i,k)
  multiplied by sigmoid gate(i,k). Outward return aggregates(k,j), weighted by
  bias/gate(k,i). Two dense contractions avoid storing a full cubic score tensor.
- Parameter inventory:5,880,961, only9,760 above the5,871,201 parent.
- Alternative: averaged pair context is redundant or oversmooths; a small
  pointwise FFN gain may have been optimization rather than relation reasoning.
- Cheap falsifier: algebra/mask/permutation tensor tests; native optimizer
  preflight requires all four return gradients finite and positive after warmup.
- Decision changed: retain or close efficient direct pair communication, not a
  slot count, width, distance or chemical-feedback retry.

### B — query-conditioned triplet attention

- Same baseline deficiency, but tests whether information selection depends on
  the target pair. Query(i,j) matches key(j,k) inward/key(k,j) outward, divided
  by sqrt8; add the same third-pair bias and multiply the same sigmoid gate.
- Parameter inventory:5,889,409,18,208 above the parent. Cubic attention storage
  and work are explicit costs, not implied negligible by the parameter count.
- Alternative: query specificity can overfit reused development molecules and
  increase runtime without improving generalization; TGT's geometric training
  could be essential for its own improvement.
- Cheap falsifier: attention reduces algebraically to A with zero query/key;
  remote optimizer preflight checks memory, timing and all four gradients.
- Decision changed: whether query-conditioned relation selection warrants its
  cost relative to both the frozen parent and A. No automatic combination.

## Frozen comparison contract

Reuse [qualified local reference](../pcqm_gptrans_local_control_100k/reference/reference_bundle.json),
including actual repository evidence and portable target transform; no scalar
reference. Independent strict comparisons may differ only in architecture
identity. Preserve fixed cross-platform100K assets, train[0,100000),
development[100000,150000), atom9/bond3/SPDcap20 consumed inputs, seed42,
FP32/noTF32, BS128/accumulation1/drop-last32, deterministic algorithms,
normalized Gap-L1, fixed train100K mean/sample-std transform, AdamW lr1e-3/
wd0.05/single-group/foreach=false/fused=false, clip1, original dropout/drop-path
0.1,60passes/warmup4/cosine/min1e-6, EMA0.999 each step. Exactly46,860 updates
and5,998,080 presentations per arm. Initialize from retained **scratch initial**
tensors, never trained weights. Added projections use forked seed42 and zero
return; CPU packaging only, no local model execution.

Retain the benchmark's0.003eV material nomination gate and positive paired-row
bootstrap lower bound. This policy gate is not a universalV5 constant or an
estimate of training variation. Per-arm native outcomes remain conservative;
below-gate/inconclusive observations do not become promotion truth merely
because their trace/reference pair is Replay-ready.100K selection cannot prove
500K transfer, convergence or official-validation ranking.

## Cost, durability, acceptance and custody

Estimate5notebook wall hours/10allocatedT4-hours; ceiling7wall/14allocatedT4.
Use two isolated workers, one visibleT4 each, with independent RNG/model/EMA/
optimizer/sampler/checkpoints. Native optimizer-inclusive preflight requires
estimated training<=5hours and>=15% VRAM headroom; otherwise preserve the
failure without changing the scientific recipe. Account quota/billing are not
API-certified. If either worker fails, preserve both outcomes; no silent
fallback, repeat or successor is authorized by this protocol.

Use native atomic checkpoints each epoch and independently retrievable chunks
every10epochs, full RNG/optimizer/resume state and60canonical observations.
Record per-insertion update RMS/weights/last-batch gradients, live/EMA curves,
selected predictions/targets/source indices, all hashes, source/runtime/
initial/transform identities, observed roles and actual native cost (or explicit
unknown measurements). Independent saved-output acceptance and terminal/RML
owners must bind diagnostics and actual complete candidate/reference Replay
pairs. Prospective planning is not terminal Replay admission. Do not invent
missing artifacts, labels, cost or protected-role access.

Existing Luna B monitors only the reconciled returned ID/version on a30minute
heartbeat: healthy silent; one durable terminal/fault event to existing A.
No new chat/cron, A model override or desktop control. A accepts both arms and
interprets results before any separately authorized next action.

## Reuse boundary

New mathematics lives only in `molgap.gptrans_triplet_communication`. Extend
typed native dispatch/diagnostics without editing frozen earlier addon modules.
Reuse `freeze_followup`, `prepare-release`/`check-release`, Kaggle bootstrap and
accelerator adapter, native trainer/acceptance/terminal and server A/B store.
