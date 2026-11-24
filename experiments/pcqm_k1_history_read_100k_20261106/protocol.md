# Normalized associative history — prospective100K contract

2026-11-06 JST. User authorized this Kaggle1 pair as one of two slots/four
actual trained arms. Both start from original frozen random K1 tensors,using
the accepted FP32 Muon/AdamW recipe. No baseline retraining,existing-job changes,
protected roles,automatic successor or scale-up.

## Evidence, distinction and falsifier

Desktop/server RML and targeted audit search found no completed version of this
current-versus-history pair. Server evidence at origin/molgap-server commit
5d9eb717d45feda46fda3acee0c264dcbbc0cfa1,
experiments/pcqm_k1_linear_attention_100k/decision.md, closed the previous
current-layer linear-attention replacement: terminal regression0.398949meV,
early gains lost; frozen100K weights regressed2.953425meV on another internal
development cohort. The latter was not500K training. The previous operator
replaced original slot mixers and used AdamW; those findings are context here.

This pair preserves all original mixers and compares two newly trained light
side paths under Muon. Its hypothesis is that normalized within-forward layer
history improves the terminal result over the same current-only path. It does
not presume history cures the older model's late-generalization failure.
The accepted Muon reference and cost are retained at
../pcqm_k1_muon_100k_20261106/decision.md and cost_and_scale_review_20261106.md.
No0.0901 whole-CoAtGIN result is treated as a module-level gain.

CoAtGIN source reviewed: https://github.com/xfcui/CoAtGIN/blob/main/model/modify.py,
AttMessage. It adds historical KV but normalizes with current-layer K. This
adaptation instead accumulates both KV and K consistently. It uses ELU+1 rather
than the source's exponentiation and is not presented as an exact reproduction.

## Frozen mechanism

At PRE_MESSAGE of local layers3,6,9,read nodes before their existing local
convolution. Original edge updates,convolution/residuals,BN and later slot mixers
retain their path. Shared affine-free LN192 plus one bias-free Linear192→192
produces q/k/v,each4heads×16channels. Positive feature map is ELU+1 on q and k.
Per graph/head, compute S=sum_nodes(k outer v) and Z=sum_nodes(k).
Read r=qS/max(qZ,1e-6),flatten64,project64→192,dropout.05,and add to nodes.

`current_read`: use only each layer's S,Z. `history_read`: carry and add S,Z
from preceding tapped layers in the same forward,with gradients through history.
Reset at layer3; clear after layer9. Never retain across batches or molecules.
Both arms share projection weights across all three taps,so accumulated states
use a common coordinate system. The shared module is serialized through its
registered taps; optimizer parameter iteration deduplicates the tied parameters.
Unique learned addition:49152 parameters in each arm; no learned LN affine terms.
Both final projections start at1e-3 times identical normal Torch initialization,
not a claimed function-identical baseline. Same seed42/addon RNG isolation.
The only paired variable is retaining preceding layer S,Z.

Reuse existing substrate PRE_MESSAGE bindings and segmented per-graph reductions;
no NxN attention matrix,new scheduler,shared trainer or RML schema.
Sparse graph membership is prepared on CPU; no generated3D is constructed.

## Recipe and resource boundary

Experiment.json pins the original k1-post-bn100K recipe,data and random
initialization,with registered addon and k1-muon-adamw-fp32-v1 only.100K train,
50K internal development,40passes/31240updates/3998720presentations,
BS128/drop_last,single forward,seed42,FP32/noTF32,native Torch2.11,
LR4e-4/WD1e-5/clip1,cosine40→1e-6/no warmup. Best-development-Live selection,
EMA.999 tracking without Clean-BN selection. Retain inherited checkpoint/resume,
profiling and standard epoch-only logs. Official validation/tests unauthorized.

One real worker per T4,one T4x2 session,18000wall-second ceiling,
36000allocated T4-device-seconds,cleanup180s. Expected3.5–4.5h is an estimate
from retained Muon runtime,not measured performance. Luna owns slot/status
checks; allocation and account-quota hours remain distinct. Wall pauses retain
state; this contract does not authorize automatic additional sessions.

## Acceptance and decisions

Before diagnostics publish the standard prospective records. Use existing
source/config/data freeze,remote receipt,native repeat/resume proof,full trace,
aligned finite50K predictions,selected model,atomic resumable state,actual native
cost and observed role audit. Retrieve only the strict required allowlist.
Both trajectories require independent complete replay with no exclusions.

History mechanism gate: history-minus-current selected Live MAE<=-0.003eV,
paired2000-bootstrap95percent upper<0,negative final and last-five deltas.
Report both against retained Muon0.1325206676620245eV; adoption additionally
requires>=3meV improvement and a qualified strict retained-reference comparison.
If unavailable,leave that comparison pending rather than training a baseline.
Require<=2x original model parameters; report measured native step/process/
allocation/memory separately. Row uncertainty is not seed uncertainty.

If history loses current,close this fixed history mechanism; if both lose Muon,
do not promote a relative win. If current wins,record current-only side-path
evidence without reopening the prior replacement's negative decision. Inspect
full curves before calling faster fitting an exposure or generalization cause.
Numerical/execution failures remain truthful INCONCLUSIVE,not rescued metrics.
Use existing terminal closure for both arms and its single RML rebuild,write
attribution before another module,and route reviewed adopted positives to
desktop or complete negatives to archive. No automatic500K/full release.
