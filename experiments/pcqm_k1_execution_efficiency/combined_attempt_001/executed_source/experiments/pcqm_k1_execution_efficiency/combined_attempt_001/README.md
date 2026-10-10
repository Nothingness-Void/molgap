# Joint execution diagnostic

Desktop continuation, NO_TRAIN. Prior isolated layout, optimizer and TF32 gains
must not be added. Test their interaction with the existing K1 factory/owner step
and scratch probe; no new trainer or remote job. Original payloads stay immutable.

Same accepted4096 TRAIN probe, first1920 executed members, frozen seed42 initial
state, BS128, normalized Gap L1, nine layers, no EMA/teacher/architecture change.
Normalization comes from the retained owner asset, never new role statistics.
No development/official/test predictions or labels. Discard all scratch weights.

Cases: original baseline/transfer controls, CPU layout, fused AdamW plus layout,
and fused AdamW plus layout plus matmul-only TF32. Match layout and barrier policy
for the isolated optimizer and precision increments; do not attribute the original
baseline's barrier difference solely to either mechanism. No AMP/cuDNN changes.
Every optimizer variant is a separate contract even if a short comparison is exact.

Three scratch-update comparisons retain loss, raw/clipped gradients, parameters,
buffers, optimizer state and RNG. Four balanced timing windows, warmup3 and
measured12 steps each; context/transfer/finite guards/host loss/RNG included.
No learned endpoint, frozen-checkpoint MAE or full trainer resume qualification.

Local RTX5060 only when two idle observations pass. CPU4threads, whole worker
cooperative ceiling600s; no persistent monitor or automatic retry. Record wall
and process CPU independently, GPU busy unknown. All cases must complete with
finite positive windows; failed cases are not a positive speed certificate.
Pure FP32 controls must remain exact; numeric errors are tensor errors, not eV MAE.

Nominate a combined contract only if its matched windows justify it; quality and
native T4/A100 qualification remain separate and unauthorized here. Terminal
attribution is required before another mechanism. No production/default switch.
