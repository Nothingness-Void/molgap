# Exact-hop content protocol — 2026-10-24 JST

User approved the reported100K dual-arm plan and Kaggle1 execution. Desktop
owner codex/exp/k1-hop-shell-100k starts from verified desktop ad4481f2.

## Scientific delta

Both arms preserve the original K1 real-bond GINE, eighteen local BN modules,
RWSE16, one-slot global mixers, mean readout and prediction head. Add one
PRE_MESSAGE node residual at layers3/6/9. Each has LayerNorm192, a bias-free
192-to64 content projection,64 learned diagonal decay logits, LayerNorm64 and
a zero-initialized bias-free64-to192 return. Both arms have identical parameter
keys, initial values, widths, normalization and four recurrence updates.

For each receiving atom v, S_k contains atoms at exact undirected real-bond
shortest distance k, k=1..4. Exclude self, padding, disconnected components and
other molecules. Derive topology on CPU before input transfer; no coordinates,
virtual chemical bonds, shortest-path attention bias or random walk sampling.
Batch-local topology is shared across the three residuals and released on exit;
there is no new persistent graph dataset or checkpoint buffer.

Let x_k be the sum of projected nodes in S_k divided by the SAME union count
max(1,|S_1 union ... union S_4|). Do not normalize each shell independently.

- hop_mixed: replace each of the four x_k by (x_1+x_2+x_3+x_4)/4.
- hop_ordered: retain x_1,x_2,x_3,x_4.

Both apply s <- lambda*s+(1-lambda)*x from distance4 down to1, s initially0,
with epsilon-bounded sigmoid diagonal lambda initialized0.5. Both use the same
output normalization/residual return. The primary contrast is retention of
distance-partitioned content conditional on this encoder, not absolute module
benefit over original K1. It also does not separately identify distance labels,
ordering and learned decay effects. No GRED expressivity guarantee is claimed.

## Frozen training and roles

Fixed PCQM4Mv2 pure2D train[0,100000), internal development[100000,150000),
seed42, original full initial tensor state8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd.
Same addon initialization for both arms, no pretrained/teacher initialization.
K1/3 accelerated shared family trainer, physical batch128/drop_last, one
forward/update, FP32/noTF32, AdamW4e-4, WD1e-5, betas0.9/0.999, eps1e-8,
clip1, cosine40passes/minLR1e-6, no warmup. EMA0.999 tracked only for compatible
state retention; best-development LIVE is the sole selected endpoint. No EMA,
Clean-BN or posthoc alternative selection. Standard epoch-only logging.
40passes/31240updates/3998720 gradient presentations per arm,50000 finite
aligned development predictions. Existing development use is acknowledged.
Official validation/test/common/OOD/P8-hard remain sealed; no3D generation.

## Observations and scientific decision

Retain all epoch curves, LR, step/exposure, selected/final states and actual
native costs. At passes10/20/30/40 evaluate the same first4096 TRAIN rows in
Live/eval mode without gradients or buffer/RNG mutation. Record those as train
diagnostic MAE, not development, selection or extra gradient presentations.
Same consumed development roles may be summarized by graph diameter<=4/>4,
cycle rank0/1/2/>=3 and conjugated-edge fraction0/(0,0.5]/>0.5. Small groups
remain descriptive; no slice-optimized model choice or threshold adjustment.

Primary: MAE(hop_ordered)-MAE(hop_mixed). Nomination requires improvement at
least0.003eV AND paired-row bootstrap95% upper bound<0 (2000draws,seed42).
Row uncertainty is not training stochasticity. Otherwise close the ordered
contrast negative under this recipe, without blind cutoff/width/LR/seed retries.
If A wins, it does not establish A's absolute benefit over original K1.
Any future absolute claim needs an existing qualified matched original-K1
reference; missing comparison does not authorize baseline retraining.

After strict training acceptance, a bounded local NO_TRAIN discriminator may
replace ordered contents by the same four-slot mixture at the selected B
weights, after reproducing original predictions within1e-4eV. Evaluate the
same50000 consumed development rows once. Weights/BN buffers remain fixed;
no model selection. Budget30minutes local CPU/desktop GPU. This measures
frozen reliance on shell partition, not training-time causal gain. If original
reproduction fails, stop and retain the blocker; no tolerance relaxation.

## Cost, release and closure

One private Kaggle1 T4x2 slot, one arm/device. Expected4–5 nativeT4device hours
total is an estimate, not measured cost. Ceiling8 nativeT4device hours,
14400s Python allocation wall,180s cleanup reserve. Queue/CPU unknown remain
unknown. The all-arm native repeat/serialized-resume gate precedes training.
Additional resource qualification compares the same-batch optimizer-inclusive
step to original K1 under the same native runtime, at most25% overhead per
arm; its short probe is not a scientific baseline training arm or epoch-speed
claim. Resource failure stops formal training and records NO_TRAIN/cost truth.

Reuse registered scientific input, source closure, submission adapter, existing
worker, checkpoint/resume, logging, selective retrieval and accept_run --strict.
Only required final output/native proof/execution and small diagnostic receipts
are retrieved after terminal status and explicit size review, on D:. Never bulk
download history chunks, environments or cache copies. Each arm independently
requires complete strict replay and exclusion_reasons=[] before the pair is
called replay-ready. Missing evidence stays pending. No automatic retry,
continuation,500K/full, adoption or official-role release follows this plan.
