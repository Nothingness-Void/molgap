# CPU falsifier,local qualification and submission — 2026-11-06 JST

The standard compiler published both prospective records before diagnostics.
Frozen source aede1f674ca1d5b84a5f65d2ce025f790c07aea1;immutable Spec
350136aeee02349dcb10bdfd89c5e416c5e799d4a581cf00a72b36278e68a30d.
Only the scientific addon/catalog and experiment records are new. Training,
state/IO,source,roles/cost,native qualification,submission and RML are reused.

The declared training-only CPU falsifier passed without changing its thresholds:
1024evenly spaced training graphs,29636edges,all1024with nonconstant incoming
EC values. Twelve observed degree-pair groups had conditional EC variation.
EC2/EC3 nonzero fractions0.8756917/0.7801660. Median topology batch128 compute
0.002462s versus the0.05s ceiling;whole diagnostic0.514376wall/0.5CPU-process
seconds,excluding interpreter startup. No development quality or labels were
used for the screen;no GPU or optimizer update was used.

The first wrapper invocation rejected an experiments/ script because its
--script route is platform-confined;the scientific diagnostic had not run.
The single actual diagnostic used project Python with this checkout's explicit
PYTHONPATH and CUDA_VISIBLE_DEVICES=-1. That invocation error is retained.

Luna ran each existing same_recipe sandbox once. Degree gate passed. Expansion
gate passed repeatability,finiteness,tap execution and parameter-perturb/fresh-load
checks,but ec_gate_3.gate.0.weight had zero first-batch gradient:the shared tiny
fixture is a3-node and a2-node chain,whose third-hop frontier is identically zero.
The original FAILED report remains unchanged. The model or validator was not
modified to make an absent fixture signal nonzero.

The missing coverage was probed once on128actual authorized training topologies,
source_idx0–127,using the frozen model loader and family forward. This used
prediction.sum().backward(),no labels,development selection or optimizer update.
EC3 was nonzero on3187/3924edges;all10registered addon parameter tensors had
finite nonzero gradients,including all16elements of the previously zero weight.
CPU wall8.431s. This establishes connectivity under actual topology,not native
training/resume or accuracy. Already passed repeat/reload checks were not rerun.

Raw reports,retained verbatim under
`D:/molgap-artifacts/reviews/k1-ec-gate-20261106/`:
cpu_viability.txt,cpu_viability_direct.txt,degree_gate-sandbox.txt,
expansion_gate-sandbox.txt,expansion-gate-real-training-topology-gradient.txt.

Official submit_workflow returned SUBMISSION_BOUND for Kaggle3 kernel143052037/v1.
STATUS.md and the D: launch receipt retain actual source/Spec/package/version.
Luna owns status checks. Native preflight still decides actual T4 repeat/resume
qualification and formal workers retain complete logs,costs and role evidence.
No saved outputs were bulk-downloaded,no shared trainer/RML framework was added,
and no existing job or scientific conclusion was modified.
