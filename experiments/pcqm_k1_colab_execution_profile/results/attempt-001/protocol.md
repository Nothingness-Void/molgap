# Frozen K1 execution-only profile — 2026-10-07

Question: what costs drive K1 two-pass training, and do selected-state
consistency gradients conflict with supervised gradients on training members?
Authority: user requested local notebook preparation, browser upload, Colab
A100 execution and private Google Drive persistence while Kaggle trains.

Reuse accepted native500K loader/factory, frozen forward/objective and FP32
settings; shared atomic IO/hash/RML planning. K1 parameters3658817, selected
epoch49 checkpoint, source/normalization unchanged. Scratch optimizer steps are
execution probes, discarded after each case; no replacement model is selected.

Sample4096 unique train members drawn uniformly without replacement from
[0,500000) with numpy default_rng(20261007), in drawn order; physical batch128.
Local CPU staging decodes only ten training shards, strips all geometry before
serializing the sampled graphs. No development, official validation, test,
common/OOD or challenge roles. Capture source/parent shard/row/target identities.

Matched FP32/noTF32/deterministic runtime: Python3.11, Torch2.4.1 CUDA12.1,
PyG2.6.1, OGB1.3.6, NumPy1.26.4; no torch-scatter/sparse extensions. AdamW4e-4,
weight_decay1e-5, foreachFalse/fusedFalse, clip1. Each case restores the same
selected checkpoint and seed42: single-forward workers2, two-forward workers0,
2 and4; five warmup and24 measured steps. No scheduler/epoch inference.
Single-forward normalized L1 is an execution counterfactual with different
dropout/BN/loss behavior, not an equivalent predictor or MAE ablation.

Secondary: eight separately synchronized two-forward steps (discard first2)
split loader/H2D/forward+loss/backward/clip/optimizer. Three extra steps supply
the CPU/CUDA operator table; profiling overhead is separated from throughput.
Four batches at the original selected state measure supervised versus
0.1*disagreement parameter-gradient norm ratio and cosine, without optimizer
updates; scratch BN buffers may update and are discarded. Post-hoc selected
state signals cannot establish trajectory-wide interference or causal MAE harm.

Worker ceiling1200s actual A100 allocation, setup/installation reported
separately. Native T4 speed/billing, development/serialization costs, full epoch
time and model quality are unresolved here. Atomic per-case checkpoints of
analysis progress and final hash manifest persist to Drive. Resume retains
completed artifacts; an unknown run is reconciled before rerun. Outcome is
NO_TRAIN or an honest incomplete blocker, never training replay-ready.

