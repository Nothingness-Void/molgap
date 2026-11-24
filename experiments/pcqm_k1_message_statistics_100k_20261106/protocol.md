# Gated-message statistics — prospective100K contract

2026-11-06 JST. User authorized this Kaggle1 pair as one of two slots/four
actual trained arms. Both start from the original frozen random K1 tensors;
both use the accepted FP32 Muon/AdamW implementation. No reference retraining,
change to existing jobs,protected roles,automatic successor or scale-up.

## Evidence review and falsifier

Desktop/server RML queries and the architecture audit found no completed version
of this mean-versus-dispersion pair. Original SSMA retained the sum but regressed
2.067051meV and incurred48.977153percent measured optimizer-step overhead:
../pcqm_k1_local_mixing_clean_aux/kaggle3_accuracy_reconciliation_v1/terminal_decision.md
and attribution.md. That FFT/joint residual differs from channelwise moments;
its failure does not prove moments useful or useless. The extremal pooled
readout regressed3.333meV (../pcqm_k1_extremal_readout_100k/decision.md);
this intervention acts on real incoming bond messages, not graph pooling.
Degree balancing modified message scaling; neither arm rescales the original sum.

The accepted K1/Muon40-pass pair is the common optimization evidence:
../pcqm_k1_muon_100k_20261106/decision.md, selected Live0.1325206676620245eV.
Its runtime and evidence are retained; no baseline is retrained. Historical
AdamW results are context, not causal estimates under Muon.

Hypothesis: local message dispersion provides useful information beyond a
parameter-matched first-moment auxiliary decode. Falsifier: one matched complete
mean/std pair, not an entire PNA reimplementation or width/degree-scaler grid.
PNA source: https://proceedings.neurips.cc/paper_files/paper/2020/file/99cad265a1768cc2dd013f0e740300ae-Paper.pdf.
Its full multi-aggregator/degree-scaler result is not an isolated std claim.

## Frozen intervention

Only layer6 changes. Capture the actual post-gate message tensor using PyG's
existing message hook, then augment the untouched original additive aggregate
through the existing POST_AGGREGATION substrate. No neighbor cap or sampling.
For each receiving node/channel, mean=sum/d and variance=sum((message-mean)^2)/d.
The std arm uses sqrt(max(variance,0)+1e-6)-sqrt(1e-6), yielding zero for
zero-dispersion nodes. Empty receivers remain zero. Reuse the existing segmented
aggregation plan for the centered-square reduction; topology preparation uses CPU.

`message_mean`: project(mean). `message_std`: project(std). The auxiliary map is
Linear192→64,SiLU,Linear64→192 without output bias. Identical seed42 initialization
and parameter count:24640 added learned parameters per model. Initialize final
projection at1e-3 times its normal Torch initialization, identically in both arms;
this is a small nonzero start, not a claimed function-identical baseline.
Original local sum,residuals,BN,slot mixers,head and all transported base tensors
remain unchanged. The only paired variable is the statistic supplied to this map.
The mean arm tests additional decoding/first-moment scaling, not new distribution
information; std winning this control does not prove universal sufficiency.

## Recipe, roles and resources

Use experiment.json's frozen k1-post-bn100K preset with only its addon and
optimizer declarations changed: pure2D OGB atom9/bond3/RWSE16,100K train/50K
internal development,seed42,40passes/31240updates/3998720presentations,
physicalBS128/drop_last,single forward,FP32/noTF32,native Torch2.11,
k1-muon-adamw-fp32-v1,LR4e-4/WD1e-5/clip1,cosine40→1e-6/no warmup.
Select best-development-Live; track EMA.999 without Clean-BN selection.
Retain preset checkpoint/resume,profiler and standard epoch-only logging.
No official validation,test-dev or test-challenge access.

One actual worker per T4,one T4x2 allocation. Cap18000wall seconds and36000
allocated T4-device-seconds,cleanup reserve180s. Expected3.5–4.5h is an estimate
from the retained Muon process11177.999s plus addon/setup headroom,not a
measurement. Luna owns quota/slot/status observations; scheduler truth wins.
Wall pauses retain state; extra sessions need a separate user decision.

## Acceptance, interpretation and terminal route

Use the existing prospective→source/package→native qualification→trace/artifacts
→role/cost→strict acceptance→terminal RML path for each arm. Retrieve only strict
required manifests,selected predictions/weights,resume,proof and cost inputs.
Require exactly40 completed passes and the full frozen exposure.
Keep source,recipe,data,actual kernel/version,atomic state and role-open provenance.
Each arm must independently qualify with complete replay and no exclusions.

Predeclared mechanism gate: std-minus-mean selected Live MAE<=-0.003eV,
paired2000-bootstrap95percent upper<0,negative final and last-five matched deltas.
Report every arm against retained Muon's selected endpoint as a contextual check;
material adoption additionally requires a qualified strict retained-reference
comparison and>=3meV improvement over that reference. Missing qualification is
pending,never implicit baseline retraining. Both parameter counts must be<=2x
original; native step,process,allocation and peak memory are measured separately.
Row intervals do not measure seed uncertainty. Compare complete curves and
compatible train/dev quantities before claiming fit or exposure explanations.

If std loses mean, close this fixed dispersion implementation. If both trail
Muon, do not advance merely because one wins their pair. If mean improves,
interpret it as first-moment/decoder evidence, not proof of useful dispersion.
Numerical/execution failures close truthfully; no NaN masking or gate relaxation.
Write attribution beside the decision before another module,close both RML
trajectories with the existing owner,rebuild once there,and route adopted
positives to desktop or complete negatives to archive after reviewed closure.
