# K1 SOAP versus Muon: prospective100K contract

User authorized Kaggle3 paper-inspired submissions and explicitly instructed
autonomous implementation/submission without waiting for further approval.
This independent question starts from committed desktop7add4c09. Shared SOAP
support requires its own reviewed infrastructure commit before source freeze;
no existing running/frozen job is modified.

## Evidence and research question

Desktop/server RML and targeted source/audit search found no SOAP run or registered
SOAP optimizer. The accepted local K1/Muon100K pair gives0.132520668eV versus
AdamW0.141294461eV,one seed; measured process wall2.03x. Its evidence and exact
limitations are in ../pcqm_k1_muon_100k_20261106/decision.md and
cost_and_scale_review_20261106.md. Luna observed Muon500K still RUNNING/latestv1;
no completed500K transfer finding is used to justify this screen.

SOAP original method https://arxiv.org/abs/2409.11321 and author implementation
https://github.com/nikhilvyas/SOAP/blob/main/soap.py motivate Adam updates in the
changing matrix-preconditioner basis,not Muon orthogonalization. Freeze the exact
reviewed upstream revision/license and any deviations with the implementation.
The related study https://arxiv.org/html/2607.02499 benchmarks3D NequIP/Allegro
potentials and five seeds; it is not2D PCQM Gap evidence. Its4.9/5.8x A100
time-to-force-error claims are not K1/T4 step or epoch predictions. No SOAP-Muon
hybrid and no claim that matrix conditioning is an established K1 bottleneck.

Question: does this specified SOAP recipe improve the complete100K terminal
endpoint/cost tradeoff over the currently accepted Muon recipe? The two methods
are fixed recipes,not independently optimized optimizer families. One matched
fresh Muon control is intentional: previous retained-only cross-run comparison
was blocked by actual Torch/CUDA build drift. It measures current same-job
runtime/endpoint rather than silently treating a historical score as a strict
control. Do not also train another AdamW baseline.

## Frozen optimizer intervention

Both models are original neural_atom_k1/3 with no addon,identical original random
initial tensors,data,seed,batch,sampler,loss,LR/WD,clip,schedule,EMA and selection.
Use the existing Muon hidden Linear/Attention matrix classifier. Only those
matrices change update algorithm;head,embeddings,one-dimensional and other
parameters retain the same native AdamW remainder in both arms.

`muon`: unchanged registered k1-muon-adamw-fp32-v1.
`soap`: registered k1-soap-adamw-fp32-v1 after its independent implementation
review. Matrix parameters:LR4e-4,WD1e-5,betas(.95,.95),shampoo_beta.95,
eps1e-8,correct_biasTrue,precondition_frequency10,max_precond_dim2048,
merge_dimsFalse,precondition_1dFalse,normalize_gradsFalse. Remaining AdamW
parameters retain baseline(.9,.999)/eps1e-8. Keep the author's order of
projection,moment,basis update and decay. These algorithm/hyperparameter
differences are the declared optimizer-recipe intervention,not solely an
isolated covariance effect.

All numerical work remains FP32/noTF32. Remove the upstream implicit FP64 retry
rather than silently changing precision. Expose eigensolver failure to native
qualification. Preserve the author's first SOAP call:initialize preconditioner
and skip that call's hidden-matrix update. The AdamW remainder still updates;
do not relabel this as identical per-parameter update counts. Both kernels
consume31240optimizer calls/3998720presentations;actual parameter steps/state
remain inspectable. No model parameter or inference path is added.

Reuse existing optimizer/session/scheduler/EMA/atomic checkpoint/resume/output,
role/cost,source/receipt,acceptance and RML owners. No experiment-specific trainer.
Required local verification is the affected real kernel/resume path once;
upstream-state compatibility,FP32 and finite updates are not inferred from schema
tests. Native T4 repeat/resume and cost checks remain on the actual worker.

## Roles,exposure and resources

Pure2D OGB atom9/bond3/RWSE16,fixed100K train/50K internal development,
40passes,BS128/drop_last,single supervised forward,seed42,FP32/noTF32,
Torch2.11,LR4e-4/WD1e-5/clip1,cosine40→1e-6,no warmup,
best-development-Live selection,EMA.999 tracking without Clean-BN selection.
Retain inherited checkpoint/resume,short profiler and standard epoch-only logs.
Official validation,test-dev/test-challenge are unauthorized.

One actual worker per T4,one T4x2 allocation;6h wall/12T4-device-hour ceiling,
180s cleanup reserve. Existing Muon process is11177.999s;SOAP runtime and memory
are unknown. Budget is a bound,not a speed prediction. Record preconditioner
state size,peak/native step timings,process/allocation wall and all retained
checkpoint/proof costs. Do not equate fewer Python kernel launches with less
mathematical work. A wall pause retains state;no automatic extra session follows.
Luna owns account/slot/status observations. Together with the EC5h slot,the
planned maximum is11account-wall-hours/22T4-device-hours,below observed Kaggle3
17.12remaining hours;the submitter rechecks quota before each mutation.

## Terminal decision

Primary gate: SOAP-minus-Muon selected development MAE<=-0.003eV,
paired2000-bootstrap95percent upper<0,negative final and last-five matched
deltas,strict same-job qualification. Report all matched trace points and actual
cost rather than promoting early fitting or claiming row intervals measure seed
variation. If SOAP ties accuracy,report its measured cost tradeoff without
retroactively changing the scientific accuracy gate. No automatic500K release.

Each arm independently needs full trace/exposure,finite aligned50000-row
predictions,selected model,resumable optimizer/basis/RNG state,native repeat/resume,
actual cost and observed role provenance. Use strict selective retrieval and the
existing terminal/RML closure. Runtime observations,replay reproducibility and
strict comparison remain separate;preserve any real mismatch without fabricated
equivalence. Write attribution beside the terminal decision before another route.
No blind LR/frequency/seed grid,SOAP-Muon fusion,baseline substitution or successor.
