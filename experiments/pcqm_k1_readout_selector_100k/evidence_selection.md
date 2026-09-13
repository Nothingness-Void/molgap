# Evidence selection — round 1

Decision date: 2026-09-13.

The frozen K1-v4 reference is a strong compact architecture because one shared
64-channel molecular bottleneck exchanges information only after local layers
3, 6, and 9. Five accepted K1 variants have already shown what not to add:
more atom slots, channel-wise selection heads, molecule-conditioned queries,
molecule-conditioned exchange strength, and a separate relation slot either
overfit or were immaterial. The first three shared the diagnostic signature of
lower training error but worse development error. This round must therefore
avoid adding another allocation branch.

## Question A: final node-set readout

K1 still compresses its final node states by a plain mean. The open RepSet
implementation from *Molecular set representation learning* maps each node
against learnable hidden-set elements, maximizes over elements, sums over set
members, then projects the hidden-set response. Its SR-GINE experiments replace
plain pooling, while the authors' V2 combines the set representation with mean
pooling. The graph search initially used eight hidden sets with eight elements,
so this experiment freezes that conservative published size rather than tuning
RepSet capacity on the development role.

The candidate keeps mean pooling and adds a zero-initialized representation
return from RepSet. This preserves exact K1 predictions at initialization and
is not a target residual or prediction fusion. A prior projected-moment readout
on GraphState was only weakly positive and below its gate; it did not test
learned hidden-set matching on K1 or under v4, so it is relevant caution rather
than a duplicate.

Primary sources:

- https://www.nature.com/articles/s42256-024-00856-0
- https://github.com/daenuprobst/molsetrep/blob/main/src/molsetrep/models/set_rep.py
- https://github.com/daenuprobst/molsetrep/blob/main/src/molsetrep/models/sr_gnn_v2.py

## Question B: selector regularization

The failed K1 allocation variants consistently benefited training fit more than
development transfer. K1's fixed one-distribution bottleneck therefore behaves
as useful regularization. `K1-TiedSelector` tests the stronger regularization
hypothesis: layers 3, 6, and 9 share one atom-selection map, but each retains
its own node values, slot seed, slot processor, and return projection. This
removes active parameters and does not add a second representation channel.

The two questions are independent and can occupy one GPU each in one T4x2 job.
Neither result may trigger another seed, role read, or scale-up automatically.
If neither passes, the coordinator must write an attribution before releasing
round 2 of the three-round authorization.

