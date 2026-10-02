# Chemical path input under corrected EMA

## Authorization and hypothesis

On 2026-10-02 the user authorized one combination test after the controller's
positive-module review. Test G1 degree initialization + G2 chemical path mean +
EMA 0.999 against the accepted G1 + EMA 0.999 reference. This changes only path
input relative to that reference. It is not a replay of the completed G1+G2
EMA 0.9999 experiment, and it does not authorize further variants.

G1 and G2 separately improved the weaker original control. Their combination
under EMA 0.9999 did not establish additivity. Correcting G1 EMA improved the
selected endpoint with exactly identical live training observations. The
remaining question is whether path information benefits corrected selection;
the alternative is redundant input or worse generalization. See the
[previous decision](../pcqm_gptrans_input_ema_100k/gpu/results/decision.md).

## Frozen comparison

Reuse the accepted cross-platform PCQM fixed100K cache and accepted CPU path
sidecar. Train rows 0:100000; internal development rows 100000:150000.
Use seed42, deterministic FP32/no TF32, physical BS128, 60 epochs, 46,860
optimizer steps and 5,998,080 presentations. Preserve AdamW single group,
learning rate/schedule/loss, train-only portable target transform, EMA 0.999,
selection rule, initialization and 5,246,817 parameters. No geometry, new graph
construction, pretraining, official validation, test-dev or challenge access.

Reuse frozen G1+EMA selected predictions (0.14423262914597987 eV), canonical
trace and qualified reference bundle; do not retrain a reference. The prior
path model and corrected EMA are reused unchanged; only their dispatch is new.
Interpret a strict paired gain above 0.003 eV with favorable row-bootstrap
interval as a material 100K finding. This prospective policy gate is not
measured seed variance. Report smaller favorable effects separately. No
automatic seed, 500K/full run or desktop handoff is authorized.

## Resource and evidence plan

One private Kaggle3 T4 allocation; one candidate isolated to GPU0 before CUDA
import. A second assigned T4 remains unused because the reference is frozen
and no second scientific candidate was authorized. Measure the full allocation
including idle devices. Estimate 3–4 wall hours / 6–8 allocated T4 hours;
hard deadline six wall hours / twelve allocated T4 hours. Preflight training
estimate must fit 4.5 hours. Retain failures; unknown submission requires
reconciliation, not retry. No blanket successor authority.

Reuse atomic per-epoch checkpoints, independent ten-epoch checkpoint chunks,
full live/EMA metrics, optimizer steps/presentations, LR, checkpoint hashes,
prediction/row/target manifests, role history, actual native cost and runtime
qualification. Before submission verify actual reference evidence and publish
the per-arm prospective plan. After completion independently accept saved
artifacts and close through existing RML. Replay-Ready is claimed only after
the actual candidate/reference pair enters the rebuilt pool with complete
trace capability; prelaunch readiness is not a terminal result. Existing server
B monitors the exact returned job every 30 minutes, silent when healthy and
notifies A on confirmed completion/actionable failure. No desktop custody.
