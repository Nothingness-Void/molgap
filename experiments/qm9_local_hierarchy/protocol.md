# QM9-30K local-hierarchy pretraining protocol

## Question

Does literature-faithful local molecular reconstruction improve a fixed 2D
GraphState9 encoder enough to nominate the mechanism for PCQM-100K transfer?
This is Track C triage. It cannot establish a PCQM result or authorize full
training.

## Immutable data contract

- QM9 processed targets and raw SDF come from the PyG-published QM9 assets.
- Split seed 42 uses 30,000 train, 3,000 internal validation, and an unmaterialized
  3,000 held-out role. The held-out labels and graphs are never read.
- Molecular graphs are rebuilt from canonical SMILES with the official OGB
  nine-channel atom and three-channel bond categories.
- RWSE16 and directed non-backtracking topology wedges are computed on the CPU.
- Twelve versioned RDKit SMARTS classes provide chemistry-defined atom-level
  functional-group membership. They are labels, not input features.
- No DFT coordinates, official PCQM role, checkpoint, external pretraining
  weight, target residual, or prediction fusion is used.

## Matched experiment

One Kunshan DCU job trains three trajectories from the same seed-42 model
initialization and cache:

1. `scratch_a`: direct Gap training for at most 40 epochs;
2. `scratch_b`: an exact repeated direct Gap trajectory for empirical platform
   dispersion;
3. `local_hierarchy`: 20 epochs of local reconstruction followed by at most 20
   epochs of direct Gap fine-tuning.

The shared encoder is a pure-2D GraphState9 with OGB categories, RWSE16,
persistent EdgeState64, WedgeState16, nine ResGated local blocks, and molecule
state exchange after blocks 3/6/9. Geometry tensors are explicit zero/masked
compatibility inputs, so no geometry information enters the model.

During pretraining, 15% of atom rows and 15% of directed bond rows are corrupted.
Training-only heads attached to the final local atom and bond states reconstruct:

- all nine original OGB atom categories at masked atoms;
- all three original OGB bond categories at masked bonds;
- twelve chemistry-defined functional-group memberships at masked atoms.

The loss is `atom CE + bond CE + 0.5 * functional-group BCE`. The inference
model and scalar Gap head are unchanged; training-only heads are discarded.

All trajectories use batch 48, AdamW `4e-4`, weight decay `1e-5`, FP32, gradient
clip 1, and cosine decay to `1e-6`. Direct-Gap phases use patience 8. Every epoch
writes an atomic last checkpoint and trace; each selected model is separately
retrievable.

## Promotion and stop rule

The candidate is only nominated for PCQM-100K when all conditions hold:

- its validation Gap MAE is below both scratch controls;
- its gain over the scratch mean is at least
  `max(0.002 eV, 2 * absolute scratch-control spread)`;
- artifacts, source/cache identity, arithmetic, finiteness, and sealed-role
  flags pass independent no-model acceptance.

A loss closes this exact pretraining objective. A win creates only one Track B
transfer candidate; it does not authorize extra seeds, full training, official
PCQM validation/test-dev, or molecular-research-server access.

## Platform and budget

An isolated CPU-only dependency directory is first built from pinned Linux
wheels uploaded with a recorded archive hash; SCNet network access is not
required. It reuses the accepted CPU Torch/PyG base without changing the shared
environment. The graph cache is then built on Kunshan `kshctest02`.
A separate 15-minute
single-batch DCU preflight verifies the exact encoder, training-only heads,
state dimensions, parameter count, finite loss/gradients, and masked geometry.
GPU training is submitted only with `afterok` dependencies on both jobs and
independently refuses an unaccepted cache or mismatched preflight.
Kunshan is used because Xi'an's old HIP/scatter runtime failed the scientific
repeatability gate. Expected wall time is approximately 1--2 CPU hours plus
3--5 DCU hours; hard limits are 6 CPU hours and 8 DCU hours.
