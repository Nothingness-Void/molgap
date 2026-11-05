# Prospective protocol draft — K1 node SwiGLU

The initial planning-only scope was superseded by the user's2026-11-06
"开跑吧" instruction. experiment.json owns necessary local qualification and
one fixed100K40-pass Kaggle3 candidate release; the original qualified comparator
is reused. There are two scientific arms and one newly trained worker. No original
reference retraining, unrelated candidate, protected-role access, successor or
promotion is authorized. A4h allocation bound (8 allocated T4-device-hours)
uses the retained1.68h process observation with qualification/headroom; this is
a new explicit execution bound, not an inherited GPTrans budget.

## Scientific intervention

Base: original neural_atom_k1/3, hidden192, nine local blocks, persistent
edge-state width64, original slot mixers and mean-pooling head. Only the node
FFN at `local_blocks[i].mlp` is replaced. Source call order remains edge update,
local convolution/dropout, residual/norm1, node FFN/residual/norm3, then the
existing slot mixer at its original layers. No normalization is replaced.

Original FFN: `Linear(192,384,bias=True) → SiLU → Dropout(.05) →
Linear(384,192,bias=True) → Dropout(.05)`.

Candidate FFN:

```text
[g,v] = Linear(192,512,bias=True)(x).chunk(2, dim=-1)
h = Dropout_hidden(SiLU(g) * v)
y = Dropout_output(Linear(256,192,bias=True)(h))
```

The hidden and output probabilities come from the original MLP's indices2 and4;
both are.05 in this K1 constructor. Gate and value matrices/biases are independent
parameter slices packed into one linear projection. There is no branch-wise
dropout, new BN, residual gate, initialization scaling or auxiliary loss.

The family loader transports the full original frozen initialization before
the addon mounts. All untouched tensors remain transported; each new FFN uses
standard Torch Linear initialization within a CPU-only RNG fork seeded42.
The CPU default generator is seeded explicitly; CUDA generators are not reseeded.
No original FFN weights are partially copied or claimed functionally equivalent.
Normal optimizer/EMA construction occurs after the existing substrate is sealed.

## Matrix and parameter budget

| Per layer | Original | SwiGLU |
|---|---:|---:|
| Matrix weights / leading matrix MACs per node | 147,456 | 147,456 |
| Bias parameters | 576 | 704 |
| Total FFN parameters | 148,032 | 148,160 |

Nine-layer totals are1,332,288 and1,333,440: +1,152 parameters. For full-model
original count P, candidate count is P+1,152. Since P>=1,332,288 from these FFNs
alone, the ratio is <=1.000865, well below2. This is source-derived arithmetic,
not a measured instantiated-model inventory. Runtime shape/parameter assertions
remain pending diagnostic authority.

Elementwise operations, activation storage and GEMM shapes differ. No native
training/inference latency, peak memory or <=2x runtime claim follows from equal
matrix MACs. If later authorized, compare measured native T4 cost at equal exposure;
do not use a language-model throughput number or CPU metadata timing.

## Reference and frozen recipe

Arm A reuses original K1 from the strict baseline in message-DropPath100K;
arm B is this single SwiGLU candidate. There are two scientific arms. A has
selected best-Live development MAE0.1412944608205557eV. It is not a trained teacher
or a modern-norm variant. Exact reference IDs and pins live in qualification.md.

Reuse that arm's recipe, not a hand-reconstructed default: fixed100K train and
next50K development,40 passes/31,240 updates/3,998,720 presentations, BS128/drop_last,
seed42/python-epoch-shuffle, FP32/noTF32/deterministic native Torch2.11 AdamW,
LR4e-4, WD1e-5, clip1, cosine40/min1e-6/no warmup. EMA.999 is tracked but the
endpoint selects best-development-Live. Normalized train-only target transform,
labels/row IDs, BN-buffer behavior, checkpoint cadence128, epoch logging and the
retained short profiler configuration remain unchanged. Candidate recipe changes
only mode/addon declaration from the reference; the recipe metadata builder owns it.

Official validation/test roles are unauthorized and must not be opened. Existing
reference role history is evidence, not a fresh role-access grant.

## Intended future scientific gate

Predeclare candidate-minus-reference selected Live MAE <=-3meV with a negative
upper paired95% interval on aligned finite50K rows,2000 resamples. Single-seed
row intervals do not measure training variation. Require original matched data,
recipe and qualified runtime/source equivalence; otherwise mark comparison
pending/inconclusive rather than repair it by implicit baseline training.

Require <=2x original full-model parameters and report native cost separately.
Material accuracy improvement does not automatically admit500K or adoption.
A complete negative closes under its contract without a blind width/WD/seed grid.
If qualification cannot establish a usable comparison, a truthful NO_TRAIN or
INCONCLUSIVE disposition is preferable to a redundant run. This draft is not
itself a final NO_TRAIN scientific verdict: the route has not been executed.

## Before any diagnostic or training

1. Resolve explicit diagnostic/training authority and any physical even-worker
   interpretation without retraining A by default; no physical layout is selected.
2. Publish canonical prospective trajectory/decision state through existing RML
   planning, with that actual authority, reference, roles and resource scope.
3. Use the existing family sandbox for finite forward/backward, nine replacement
   ownership, untouched tensors/BN/head, training-RNG isolation and restore checks.
4. Existing native T4 repeat/resume qualification and an authorized small timing
   window establish runtime compatibility/cost. No custom trainer or acceptance.
5. Freeze source/config/input and use the platform skill only after those releases.
