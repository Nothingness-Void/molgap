# Neural-Atom mixer decision — 2026-09-10

## Question

Did replacing dense atom-to-atom GPS attention with four content-addressed
latent slots improve the frozen QM9 direct-Gap screen beyond both full GPS and
a parameter-identical one-active-slot control?

## Acceptance

Kaggle2 kernel `kaseichou/molgap-qm9-neural-atom-s42`, version 1, completed all
three 40-epoch arms. No-model acceptance passed with source commit
`f2d760b584023e53b3b77aef4d9348c21e6b36f4`, cache aggregate
`80a2d256ccbbde8de5862e7ad4fd61ea7c1255a19217e590033a94cabe1c6340`,
split `62f1cdefdaec6877`, seed 42, FP32, and physical batch 128. All artifact
hashes and remote preflight invariants matched; no QM9 test or official PCQM
role was read.

## Result

| Arm | Validation Gap MAE | Parameters | Mean epoch |
|---|---:|---:|---:|
| Full EdgeState GPS9 | 0.1294568628 eV | 4,771,073 | 17.758 s |
| One active latent slot | 0.1277898252 eV | 3,658,817 | 15.125 s |
| Four active latent slots | 0.1274414212 eV | 3,658,817 | 14.758 s |

Four slots improved on full GPS by `0.0020154417 eV`, below the frozen
`0.003 eV` gate. Four slots improved on the parameter-identical one-slot arm by
only `0.0003484040 eV`, below both the frozen `0.001 eV` gate and the repository's
observed seed-42 cross-job variation. The runtime ratio was `0.8311`, and the
candidate used about 23% fewer parameters, so resource gates passed.

## Attribution

The controlled result separates two effects. Replacing nine dense atom
attention blocks with three latent exchanges was directionally beneficial,
smaller, and faster. Increasing latent rank from one to four supplied almost no
additional accuracy, however. The useful effect is therefore global-channel
simplification rather than heterogeneous multi-slot communication. All arms
completed the same 40 epochs and selected epoch 38, while the candidate passed
memory and timing gates; incomplete execution, excessive size, or a throughput
bottleneck do not explain the miss.

The multi-slot hypothesis is not rescued by trying more slots, another latent
width, different placement, an additional seed, or a longer schedule. Those
changes target capacity even though the causal K1/K4 contrast shows that slot
capacity is not the missing information.

## Decision

The exact multi-slot Neural-Atom replacement is scientifically rejected and
closed. It receives no PCQM-100K transfer, shadow audit, seed expansion,
slot/width/placement/schedule retry, desktop/full-scale handoff, official-role
access, SCNet work, or IMS work. Its efficiency signal remains design evidence
only. One of the three post-cardinality scientific attempts is consumed; two
remain, each requiring a new all-history evidence audit before submission.
