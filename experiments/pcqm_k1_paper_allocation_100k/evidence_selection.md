# Evidence selection

## Why K1 helped

The accepted K1 result removed nine dense atom-to-atom attention operations and
kept only three 64-channel latent exchanges after local EdgeState blocks 3, 6,
and 9. This reduced irrelevant global mixing, parameters, and optimization
burden. The K1/K4 QM9 control showed that merely activating four slots under the
old equation changed MAE by only `0.0003484040 eV`; K1's benefit therefore came
from the sparse low-rank communication shape, not demonstrated multi-slot
specialization.

## Why K1-G and K1-R missed

K1-G multiplied the already formed K1 update by one molecule scalar. It could
change amplitude but not decide which atoms communicate. It worsened both
training and development error. K1-R pooled EdgeState into one extra relation
vector and returned it through the atom slot's addressing weights. It improved
training error but not development error, consistent with duplicated local bond
information and mismatched edge-write/node-read addressing.

## Missing paper mechanism

Neural Atoms defines an allocation matrix from original atoms to multiple
Neural Atoms, exchanges information among those Neural Atoms, and reuses the
transposed allocation for backward projection. The authors' released
`BaseMHA` normalizes the allocation over Neural Atoms for every original atom.
MolGap's earlier K1/K4 approximation instead normalized over original atoms for
each slot. It therefore implemented attentive slot pooling, not the paper's
soft atom grouping.

The audited primary sources are the [ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8270bf9237b7d2c9a8dfce5488f000a4-Abstract-Conference.html)
and official repository revision
`6be4cfe63ad16c3d0bf5c37a55704da17feb8f3d`. The paper's PCQM-Contact setup is
an inductive long-range link task, so its 15--28 slots, every-layer insertion,
and 200-epoch optimizer recipe are not copied into this Gap screen.

## Bounded three-attempt policy

Attempt 1 changes only the allocation direction and activates the four already
allocated slots. It retains K1's 64-channel bottleneck and exchanges only after
layers 3, 6, and 9. Attempts 2 and 3 are not preselected: a terminal causal
analysis must decide whether the next uncertainty is slot count, head diversity,
or whether paper-style grouping is mismatched to graph-level Gap prediction.
No failed attempt may be rescued by a seed, width, optimizer, or schedule tweak.

