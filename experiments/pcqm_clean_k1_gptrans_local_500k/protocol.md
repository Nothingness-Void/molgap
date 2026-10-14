# Paired candidate500K training protocol

Date:2026-10-15 JST. Owner:desktop. User authorized assessing the three proposed
model groups, submitting two500K candidates to Kaggle1, and shutting down only
after actual training progress is verified. This scope includes the minimal
shared Clean-state integration and required local/native qualification.

## Arms

| Arm | Mechanism | Frozen selected state |
|---|---|---|
| `k1_clean_ema` | K1-v4, one supervised forward, step EMA.9999; cumulative BN calibration with dropout disabled on the first16,384 training rows after each complete pass | Best50K development MAE from that pass's Clean EMA |
| `gptrans_local` | Reviewed GPTrans256-node/32-pair/12-layer degree-scaled true-bond residual,64-channel local messages, step EMA.999 | Best50K development EMA MAE |

K1 training dropout remains enabled. Clean is an independent model; calibration
changes only its tracked BN buffers and restores global RNG/modes. It never
updates Live parameters, raw EMA or optimizer. The calibration policy pins the
exact input/role/order and128 batches of128. This is inference-state calibration,
not a new gradient loss or BN update on development. Clean state, policy and
staleness are retained in the existing checkpoint owner-state component.
The selected payload also retains same-step Live/Raw/Clean states for later
frozen attribution without retraining; secondary inference needs owning scope.

GPTrans is the registered `gptrans_degree_bond_local/1` implementation, not the
unimplemented `degree_scale` or `path_bond_mean` proposals. No geometry, teacher,
pretraining, consistency, scalar readout gain or unregistered composition.

## Common data and exposure

Private Kaggle1 input `nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1`, manifest
SHA630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751.
Train source_idx[0,500000); internal development[500000,550000). The50K role is
already selection-used, not independent validation. Official validation,
test-dev/challenge, common/OOD and Track A external roles stay sealed.
Cached geometry is stripped; no geometry is constructed.

Both:seed42, transported full initial tensors, physicalBS128, no accumulation,
FP32/no-TF32/deterministic algorithms, Torch2.11.0, normalized direct Gap L1,
clip1, torch randperm seed42+pass, discard32 tail rows per pass,60 complete
passes/234,360 updates/29,998,080 gradient-training presentations. Feature-only
calibration exposure is separately983,040 row presentations at60 complete
passes; it is not added to optimizer exposure. Qualification scratch is separate.
Normalization: accepted same500K train mean5.413434769952119/sample std
1.176728647311213 eV. No trained weights or development-derived normalization.

K1:AdamW lr4e-4/wd1e-5, cosine by pass60, minimum1e-6.
GPTrans:foreach AdamW lr1e-3/wd.05, four warmup passes then frozen cosine60,
minimum1e-6. The recipe files own the exact executable values and identities.
Atomic checkpoint every128 updates plus complete pass. Preserve last cursor,
model/optimizer/scheduler/EMA/RNG, clean owner state when applicable, and selected
finite aligned predictions independently from the resume checkpoint.

## Budget and outcomes

One Kaggle1 T4x2 job, one isolated device per candidate. Allocation entry ceiling
42,900s (11h55m), including install/input admission/qualification/train/development/
calibration/checkpoints/idle/publication; reserve180s. Ceiling23.8333 assigned
T4-device hours is estimated, not a measured charge or busy time. This is below
Kaggle's documented12h GPU session limit. No accelerated precision or compressed
schedule to fit. Both native repeat/serialized-resume qualifications must pass
before either trains; K1 also qualifies Clean-BN against repeated scratch states.

The prior mean2/single-EMA job stopped under its own9h ceiling after29/53
complete passes. It is not a failed model result, a complete comparator or a
reference to retrain. New topology execution and removal of mean2 reduce work;
completion within this budget is not guaranteed. At a ceiling retain truthful
STOP_FOR_COST/matched exposure and independently durable resume, never report60.
No automatic continuation, retry, new monitor, full run or adoption.

## Scientific decision and terminal closure

Accept source/data/native/state/alignment/trace/cost/role integrity per arm first.
Per-arm complete training/RML evidence is required; reference qualification and
replay readiness remain fail-closed, never promised from a successful launch.
Historical500K endpoints are context until strict comparison readiness passes.
Cross-family same-job scores do not isolate architecture, optimizer or EMA effects.

Nomination requires an unchanged3meV material improvement with positive95% paired
row interval against an applicable qualified500K reference. Missing reference
means pending/INCONCLUSIVE, not baseline retraining or threshold relaxation.
Record clean state versus raw historical context without attributing all of a
new cross-job difference to calibration. Row bootstrap is not training variance.
Both complete endpoints may support one fixed50:50 saved-prediction descriptive
blend; no weights are fitted, model execution repeated or deployment claimed.
Write terminal attribution before any unrelated successor, close each arm via
existing family acceptance/experiment_cli/research_memory, rebuild/check RML,
then route only after the experiment decision under BRANCHES.md.
