# Frozen protocol — K1 edge-conditioned slot

## Scientific question

K1 uses one 64-channel molecular slot at layers 3, 6 and 9. Its atom-to-slot
assignment currently sees only the node state. This candidate keeps every K1
tensor and changes only the selector key:

```text
incident real-bond EdgeState -- mean + parameter-free LayerNorm --┐
                                                                   +--> slot key
node state --------------------------------------------------------┘
```

For atom `i`, the added term is the learned projection of the mean directed
real-bond state incident to `i`. The projection is zero initialized. The
candidate therefore has the exact K1 function at initialization and can learn
an edge-aware selector after optimization. It has 3,671,105 parameters: K1's
3,658,817 plus three 64×64 selector projections.

No node width, layer count, slot count, local EdgeState update, feature schema,
geometry input, pretraining, teacher, target residual, prediction fusion,
optimizer, schedule, or batch rule changes.

## Immutable v4 benchmark

Use `kaseichou/pcqm4mv2-ogb-fixed-100k-v1`, manifest SHA
`1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d`, and
geometry aggregate SHA
`bc83a4bd9a7fd7fd6fc6fd085d78ba3caab0e40f997d1932e0f344e701dd40c5`.
Train rows are `[0, 100000)` and the train-derived development role is
`[100000, 150000)`. Official validation, shadow and test roles remain sealed.

Use seed 42, direct Gap, strict FP32/no TF32, deterministic algorithms,
physical batch 128 on one visible accelerator, `drop_last`, 40 epochs,
31,240 optimizer steps, 3,998,720 sample presentations, AdamW with
`lr=4e-4`, `weight_decay=1e-5`, gradient clip 1.0, cosine schedule to 1e-6,
normalized-Gap L1, and minimum development MAE selection. Reuse the frozen
K1-v4 reference; do not retrain it in the candidate job.

## Preflight and decision

The remote preflight must verify the shared K1 initialization, zero edge-key
weights, nine finite real-bond states, three slot-assignment layers, candidate
gradient after two optimizer steps, deterministic resume, and at least 15%
reserved memory. Checkpoints include optimizer, scheduler, all RNG states,
source/runtime/data/order identities, and selected-artifact hashes; recovery
archives are published every ten epochs.

Saved-artifact acceptance recomputes the ordered 50,000-row development MAE
without constructing or executing a model. A promotion requires at least
`0.003 eV` gain over K1-v4, paired-row bootstrap upper 95% below zero, and the
memory gate. A directional or sub-threshold result closes this question; it
does not release extra seeds, scale-up, official roles, or a successor.
