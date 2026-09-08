# QM9-30K EdgeState local-hierarchy pretraining protocol v2

## Question

Does literature-faithful local molecular reconstruction improve the accepted
pure-2D EdgeState GPS9 backbone enough to nominate one mechanism for a paired
PCQM-100K transfer? This is Track C triage only. It cannot establish a PCQM
result or authorize full training.

The v1 GraphState/WedgeState protocol remains historical. It never reached
training because 254 source molecules in the selected QM9 roles failed its
mandatory canonical-SMILES round trip. V2 is a new, explicit data and model
contract rather than an infrastructure-only retry.

## Immutable data contract

- PyG QM9 processed targets and the DeepChem-published raw SDF are immutable
  source assets, identified by SHA-256.
- RDKit scans molecular structure only, without reading target values, and
  freezes the complete canonical-roundtrip-valid source pool, rejected-source
  ledger, RDKit version, and hashes before splitting.
- Seed 42 deterministically selects 30,000 train, 3,000 internal validation,
  and 3,000 held-out source identities from that frozen pool. Only train and
  validation examples are selected, materialized, evaluated, or used by the
  experiment; held-out graph construction and target access remain disabled.
- Graphs use official OGB nine-channel atom and three-channel bond categories
  plus RWSE16. Twelve versioned RDKit SMARTS classes provide atom-local
  functional-group labels during pretraining only.
- No wedge, distance, angle, coordinate, DFT geometry, PCQM role, checkpoint,
  external weight, target residual, or prediction fusion is used.
- A separate data-only acceptance process verifies source identities, role
  disjointness, all shard hashes, and sealed-role flags before any DCU job.

## Matched experiment

One Kunshan DCU job trains three trajectories from the same seed-42
initialization and accepted cache:

1. `scratch_a`: direct Gap training for at most 40 epochs;
2. `scratch_b`: an exact repeat measuring platform dispersion;
3. `local_hierarchy`: 20 epochs of local reconstruction followed by at most 20
   epochs of direct Gap fine-tuning.

The shared inference encoder in all trajectories is exactly a pure-2D
`OGBEdgeStateStructuralGPSWrapper`: hidden width 192, nine GPS layers, four
attention heads, persistent real-bond EdgeState64, RWSE16, mean pooling, and
one direct Gap head. The candidate changes supervision only; it does not change
the inference architecture.

During pretraining, 15% of atom rows and directed bond rows are masked.
Training-only heads attached to final atom and persistent-bond states reconstruct
all nine OGB atom categories, all three OGB bond categories, and twelve local
functional-group memberships. The loss is
`atom CE + bond CE + 0.5 * functional-group BCE`; those heads are discarded.

Every trajectory freezes physical/effective batch 48, one DCU, AdamW `4e-4`,
weight decay `1e-5`, FP32, gradient clip 1, cosine decay to `1e-6`, seed 42,
the same ordered role identities, and the same validation/evaluator. Direct-Gap
phases use patience 8. Every epoch writes an atomic resumable checkpoint and
trace.

## Promotion and stop rule

The candidate is nominated for exactly one paired PCQM-100K transfer only when:

- validation Gap MAE is below both scratch controls;
- gain over their mean is at least
  `max(0.002 eV, 2 * absolute scratch-control spread)`;
- independent no-model acceptance verifies artifacts, source/cache identity,
  arithmetic, finiteness, and sealed roles.

A loss closes this exact pretraining objective. A win does not authorize extra
seeds, full training, official PCQM validation/test-dev, or molecular-research
server access.

## Scale-transfer invariant

QM9 only decides whether the mechanism deserves PCQM testing; its absolute
metric is never compared with PCQM. If nominated, PCQM scale-up starts at its
own 100K S1 contract. From PCQM-100K to 1M and full data, only eligible training
row count may change. Layer count, width, physical/effective batch, device
count, features, optimizer, learning rate, schedule, precision, seed,
validation identities, evaluator, epoch/exposure rule, and checkpoint selection
remain identical. Any mismatch is a new S1 experiment, not scale-up evidence.

## Platform and budget

CPU parsing, graph construction, and acceptance run on Kunshan `kshctest02`
from fully staged immutable assets. Only after acceptance may a 15-minute
single-batch DCU preflight exercise the exact encoder and auxiliary heads.
Training is submitted only after both gates pass. Hard limits are six CPU hours
and eight single-DCU hours; no dependency job is pre-submitted before its gate.
