# K1 pre-norm100K protocol

## Question and scope

User authorized one dual-arm validation after reviewing the100K norm attribution.
Owner: desktop. Platform: Kaggle2/kaseichou, private T4x2. One seed42 attempt;
no automatic retry, monitoring, successor,500K/full training or production change.
Allocation ceiling14,400 entry-wall seconds,180 seconds cleanup reserve;
8 assigned T4-device hours maximum across the pair, expected approximately4.
Queue/billing and account GPU-quota units remain distinct from native device cost.

Does adaptive node normalization improve K1 when used inside pre-norm residual
branches, and is the previous direct post-norm substitution conclusion sensitive
to the residual-placement package? The first question has a same-job control;
historical placement contrasts retain their source/job/seed limitations.

## Two architectures

- `k1_pre_bn`:9 local blocks, each `h=x+Dropout(GINE(BN1(x),edge))`,
  `y=h+MLP(BN3(h))`. One final node BN after mixer9 and before graph pooling.
- `k1_pre_ada`: identical operations and final-normalization placement, replacing
  these19 norms with AdaRMSNorm. `alpha=0,beta=1,eps=1e-6` initially.
- Existing mixers/LayerNorms, dynamic edges, pooling and head are unchanged.
  Dropout remains0.05 throughout the original paths. Local message edge states
  are updated in the original order. This is a local pre-norm adaptation, not a
  complete PPGT reproduction. Pre-norm includes the declared final node norm;
  historical comparisons cannot attribute that package solely to one seam.
- Both arms strictly load the same retained full seed42 K1 tensor artifact
  before the statically registered norm transformation. Preserve every shared
  learned tensor exactly; initialize only changed/new norms with the declared
  deterministic zero/one tensors. Retain derived full-state hashes/artifacts
  locally. No random parameter initialization replaces transported tensors.

## Shared recipe and roles

PCQM4Mv2 pure2D OGB atom9/bond3/RWSE16, fixed100K train indices[0,100000),
50K internal development[100000,150000). Manifest and exact hashes in config,
role plan and recipes. Development is already selection-used. Official
validation/test, common/OOD and TrackA roles are forbidden. No geometry work.

Both arms:40 passes,781 updates/pass,31,240 updates,3,998,720 optimizer-batch
presentations; BS128, drop-last, no accumulation; FP32/noTF32; Torch2.11.0;
Python epoch shuffle seed42. AdamW lr4e-4,weight_decay1e-5,betas(.9,.999),eps1e-8;
clip1.0; epoch cosine40 to1e-6, no warmup. Train-only target mean5.3383002281188965
and std1.275090217590332eV. Preserve shared accelerated backend and native
qualification. No early-stop rule from partial curves.

Primary selection is best-development-live for both arms. Track EMA.999 and
evaluate it on the same already authorized development role at each pass as a
secondary state only; it never replaces Live selection or changes the gate.
Use shared epoch logs, trace/role/cost/checkpoint owners. No Clean-BN calibration.
Online train MAE is not a fixed-checkpoint evaluation and cannot prove overfit.
Retain selected and terminal full states for any separately scoped fixed-cohort
post-acceptance attribution; no guessed fixed-cohort metrics enter this run.

## Frozen decisions

Let delta=MAE(pre-Ada)-MAE(pre-BN), using aligned finite selected Live50K predictions.
Noninferiority requires delta<=0.0005eV and paired-row95% upper bound<=0.0005eV.
Material superiority requires delta<=-0.001eV and paired-row95% upper bound<0.
Use the existing paired-bootstrap owner,10,000 draws,seed42. Row intervals do
not estimate training-seed variance or establish a causal mechanism.
If neither gate passes, close this exact adaptive replacement negative under
contract; valid partial/missing evidence remains inconclusive. Numerical gains
alone do not authorize500K. No baseline retraining to fill missing references.

Compare each cell descriptively with the pinned retained Post-BN/Post-Ada
predictions and contracts. Their31,240-update exposure and role identities must
match before any comparison. Different software/source/job effects prevent a
code-only placement causal claim. The fresh Pre-BN arm is a changed architecture,
not a rerun of the old Post-BN control.

## Acceptance and durability

Before either arm trains: all-arm native repeat, actual serialized resume,
source/init/data/role and accelerator qualification. Preserve immutable source,
actual job/version, atomic model/optimizer/scheduler/EMA/RNG/cursor checkpoints,
selected Live predictions/models, full40-row trace, measured native allocation
and process costs. Every completed arm receives independent mechanical/scientific
acceptance and terminal V5/RML closure. Bind the same-job reference prospectively;
require two independently qualified entries before claiming dual replay-ready.
Download only acceptance-required files and minimal diagnostics. No inferred costs.
