# GPTrans final atom versus bond readout

## Question and authorization

On 2026-10-03 the user authorized one Kaggle3 two-arm notebook after reviewing
the plan. Test whether access to real atom or real-bond relation states improves
the final Gap readout. Virtual-node/self-pair pooling is not a confirmed defect;
this is an information-access hypothesis, not a claim about official GPTrans.

The accepted EMA correction separated selected-weight lag from live training.
Chemical path additivity, grouped decay and uniform pair-depth scaling have not
established gains over the corrected reference. See the independent
[EMA decision](../pcqm_gptrans_input_ema_100k/gpu/results/decision.md) and
[pair-scale decision](../pcqm_gptrans_pair_scale_100k/gpu/results/decision.md).
The separately running path+EMA combination is not modified or used as an
unaccepted new baseline. Final readout is distinct from the closed all-layer
persistent pair-to-node readback and K1 readout-selector experiments.

## Single-factor arms

- `degree_node_mean_readout_ema999`: replace the 256-channel virtual node with
  the masked mean of final real-atom states. Keep the 32-channel virtual self
  pair and original prediction MLP.
- `degree_bond_mean_readout_ema999`: keep the virtual node; replace the virtual
  self pair with the mean of final pair states at real directed chemical bonds.
  Exclude padding, virtual edges and nonbonded pairs. Bondless molecules use
  the reference self-pair fallback. Keep the original MLP.

Neither arm changes propagation, tensors, capacity, targets or input features.
Freeze 5,246,817 parameters, G1 accepted initialization, train rows 0:100000,
internal development 100000:150000, accepted fixed cross-platform cache,
seed42, physical BS128, deterministic FP32/no TF32, 60 epochs/46,860 steps/
5,998,080 presentations, original AdamW single group, LR schedule, loss,
train-only target transform and EMA0.999. Reuse the accepted G1+EMA reference
at 0.14423262914597987 eV without retraining. No 3D, pretraining, protected
official validation/test-dev/challenge, shadow or new 500K inference role.

## Budget and falsification

Run independent workers on T4x2, with independent RNG, optimizer and atomic
output directories. Expected 3–4 wall-hours/6–8 allocated device-hours; cap
5 wall-hours/10 device-hours against the user-reported remaining 11 hours.
Preflight training estimate must not exceed 3.5 hours, reserving time for
evaluation and artifacts. Record full allocation, including idle time.
The platform's quota reservation and queue may prevent starting before reset;
do not retry an uncertain submission or cancel the other bound task.

A material finding requires a strictly greater than 0.003 eV paired gain and
a favorable aligned-row interval under this prospective policy. Smaller gains
are reported below gate, not promoted. Row bootstrap is not seed variance or
proof of scale transfer. Compare each arm to the frozen reference, not to the
other arm as a one-factor causal contrast. No automatic extra seed, ensemble,
500K/full training or successor is authorized.

## Reused evidence path

Reuse GPTrans native V5 preflight/training, isolated Kaggle scheduler, shared
prospective release, package/release checks, saved-artifact acceptance and
per-arm terminal/RML adapter. Retain complete live/EMA traces, steps/exposures,
atomic checkpoints, independent ten-epoch chunks, prediction/row/target hashes,
runtime certificates, observed role history and native allocation cost. Bind
the actual supplied reference evidence before compute release. Replay readiness
is a post-run property requiring both actual candidate/reference pairs in the
rebuilt pool; incomplete or stopped runs remain honestly partial.

Use existing Luna B and its 30-minute heartbeat. Preserve the old job's binding
until terminal reconciliation; the new exact physical job is a separate binding
in the same conversation, not a new chat or automation.
