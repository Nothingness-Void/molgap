# Frozen K1 BN mechanism diagnostic on A100

Desktop question, authorized2026-10-08: distinguish dropout-dependent
normalization statistics from repeated-forward BN updates in the accepted
epoch49 K1 predictor. Base desktop143708eb. Prior BN calibration recovered
1.245meV; A100 profile identified two-forward cost but not accuracy causality.
This question changes buffers only, never learned weights or a training recipe.

Use the exact accepted selected checkpoint, matching frozen model source and
target transform. Reuse load_native500k_k1, predict_clean and
recalibrated_batch_norm. Reconstruct accepted predictions within1e-4eV first.
All parameters remain frozen, no gradients/optimizer or checkpoint selection.

Four fixed interventions: dropout off/on crossed with one/two forwards per
calibration minibatch. Restore the exact original state between cases. Reset
tracked BN statistics and use cumulative minibatch momentum=None; only BN and
the explicitly enabled Dropout children, functional LocalGPSBlock dropout
owner and MultiheadAttention dropout owner may have train flags. Reset seed42
for each case. Use identical16384 unique training members, NumPy seed20261008,
from[0,500000), identical order and BS128. These are the prior calibration
members; no sample/momentum/size sweep. Cumulative duplicate passes are an
estimator counterfactual, not a reproduction of historical EMA momentum or
the evolving training trajectory.

Evaluate original plus four variants on all50000 previously consumed internal
development rows[500000,550000), clean mode, aligned finite MAE Gap/eV/minimize.
Preserve per-case predictions and BN snapshots; require parameter/non-BN
invariance and original-state restoration. Packed shard staging reads backing
labels of500K training and50K development rows; calibration consumes features
only. Development labels compute metrics, previously selection_used=true.
Official validation/test/common/OOD stay sealed. Pure2D, no geometry creation.

Predeclared contrasts: off1 versus on1 and off2 versus on2 isolate dropout;
off1 versus off2 and on1 versus on2 isolate duplicate passes. Report all cases,
not just the best. Local existing paired_bootstrap_mean,1000draws seed20261008,
95% intervals; four exploratory contrasts without multiplicity correction.
A >=1meV gap and positive paired lower bound nominates a mechanism for separate
qualification. Tiny off1/off2 difference is an expected duplicate-control check,
not evidence that historical two-pass EMA was harmless. Row uncertainty is not
seed variance. These selected-development results cannot adopt a new model or
prove a training-time intervention will recover the same benefit.

Colab native A100, Python3.11/Torch2.4.1+cu121/PyG2.6.1, FP32/noTF32,
deterministic, BS128, no optimizer. Worker ceiling1200s; setup separate bounded
timeouts. CPU graph staging before accelerator. Log each stage/batches, output
atomically to private Drive MolGap/V5/runs/k1-bn-mechanism-a100-20261008/attempt-001.
Bind package SHA, source, exact notebook/run path and observed runtime. No blind
retry. Retain output hashes, process cost, predictions and diagnostic result;
disconnect/delete runtime when terminal. Close NO_TRAIN with existing RML
plan/finalize/rebuild/check; no training trace or training replay-ready claim.
