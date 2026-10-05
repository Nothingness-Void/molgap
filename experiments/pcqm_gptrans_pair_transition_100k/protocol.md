# Independent pair transition: frozen100K screen

Frozen 2026-10-05, MOLGAP-COMMON-V5-FINAL; server-owned.
User authority: "进行提交吧 kaggle2", following the recommended first candidate.
One new seed42 candidate only. No baseline rerun, new seed, geometry, teacher,
pretraining, protected-role access or automatic successor/500K training.

## Hypothesis and contrary evidence

The retained GPTrans updates32-channel pairs through an8-head GPA projection.
That static operator shape does not establish empirical rank collapse or a
causal bottleneck. Four independent nonlinear pair residuals may improve its
relation expressivity without increasing node width. The alternative is that
GPA already supplies adequate relation processing and extra pair freedom
overfits or costs more than it helps. Related negative pair normalization,
uniform damping and relative-value routes remain closed.

The evidence and primary-paper mechanism review are in the
[research report](../pcqm_v5_route_portfolio/relation_family_research_2026-10-05.md).
The local-bond intervention's small positive result does not license stacking
it here. Neither GRIT's published budget nor its scores are transplanted.

## Single changed mechanism

Before original GPA blocks3/6/9/12, apply pair + Linear64to32(GELU(Linear32to64(
LayerNorm32(pair)))). Each block has independent parameters. Return weight and
bias start at zero; accepted G1 core tensors are copied exactly, addon-only RNG
uses seed42 in a fork. There is no addon dropout or forward RNG consumption.
Only valid real-atom ordered pairs, including diagonal, change. Virtual-node
and padding rows/columns remain unchanged by the addon. Placement before
block12 permits relation changes to reach the graph-token readout.

Base5,246,817 parameters +17,024 =5,263,841. No node/pair width, original GPA,
FFN, depth, readout, degree conditioning or optimizer alteration. Masking and
parameter identity pass focused tests; actual remote preflight remains required.

## Immutable training and reference

Use accepted cross-platform fixed100K train[0,100000) and internal development
[100000,150000), direct Gap, OGB atom9/bond3/SPDcap20, seed42, FP32/no TF32,
physical BS128/drop_last/no accumulation, portable accepted train100K transform,
normalized L1, AdamW lr1e-3/wd0.05/foreach=false, clip1,60 epochs, warmup4/cosine
to1e-6, EMA0.999 every step. Exactly46,860 updates/5,998,080 presentations.
The frozen correctedG1+EMA999 accepted bundle is reused, not retrained.
Only architecture_config_identity differs in the strict prospective plan.

The material nomination gate remains0.003eV for this matching benchmark, with
positive paired-row bootstrap lower bound. It is not a universalV5 threshold,
measured training variance or a full-scale qualification. Single-seed wins
only nominate further controller interpretation under separately covered cost.

## Execution, durability and planned RML closure

One isolated T4 worker; request NvidiaTeslaT4 only. The native scheduler expects
T4x2, so count both allocated devices although only one is used. One hypothesis
and immutable reference justify no second training arm. Estimated4wall hours/
8allocatedT4hours, hard maximum7wall/14allocatedT4hours. Before optimization,
reject if optimizer-inclusive estimate exceeds6hours or memory headroom<15%.
Do not reduce batch/precision or reset schedule to fit the budget.

Reuse existing native source/arm binding, runtime calibration, applicable role
events, measured native costs, full live/EMA trace, LR/steps/presentations,
aligned retained predictions, atomic full model/optimizer/scheduler/EMA/RNG
checkpoint each epoch and independently retrievable ten-epoch chunks. First
scheduled train batch records transition input/return/output RMS; retain final
return weight and gradient norms each epoch without extra training data.

Independent saved-artifact acceptance must check actual source/recipe, exact
reference, parameter/init/config, runtime, all terminal hashes/counters,
finite diagnostics, source-index/target alignment, role and native cost. Reuse
the terminal transaction, then require candidate and frozen reference actually
share a complete comparable Replay pool pair. Planning is Replay-capable, not
already Replay-Ready. Missing evidence stays incomplete; never manufacture it.

## Separate post100K NO_TRAIN portability action

The [frozen audit action](audit_action.json) is packaged before100K release.
After independently accepted terminal100K only, bind its selected model hash,
source/init/transform and predictions plus the accepted G1 EMA999 portability
reference. Reproduce100K predictions first, then use the existing fixed500K
internal development[500000,550000), all50K in canonical order, FP32/BS128,
no transform refit, optimizer, parameter update or checkpoint selection.
This is a separate prospective NO_TRAIN stage with its own role/cost/trace
and idempotent release, not part of this GPU screen or a transfer claim.
Its candidate checkpoint is unavailable before terminal; release stays gated.
No500K training-prefix/official/test role is opened. A negative or incomplete
100K result may close before this action; no audit failure retrains100K.
