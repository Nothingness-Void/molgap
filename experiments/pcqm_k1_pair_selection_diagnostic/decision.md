# Frozen PairToken selection diagnostic — 2026-09-22

## Question and boundary

Would the accepted PairToken's learned layer-6 assignment support a dynamic
top-pair sparse approximation? The protocol and four frozen input hashes were
fixed before execution. This was an explanatory frozen-checkpoint evaluation
of all 50,000 official-train-derived development rows, not training or a new
model comparison. Official validation, shadow, and all test roles remained
unread.

## Acceptance

Kunshan job `122739494` completed in 4m57s with exit `0:0`. It wrote ten
independent, hash-bound 5,000-row chunks. Offline no-inference acceptance
verified every chunk SHA-256, finite metric, exact row coverage, the four input
identities, strict role flags, and an independent recomputation of the
prospective gate. The model's frozen predictions reproduced with maximum
absolute deviation `1.43e-6 eV`; maximum chunk MAE deviation was `1.49e-8
eV`. Compact acceptance is `results/acceptance_122739494.json`; the complete
numeric chunks and logs are retained in ignored
`platforms/_records/scnet/k1_pair_selection_diagnostic_122739494/`.

The initial job `122739196` failed before producing a chunk because Kunshan's
deterministic GPU build lacks a `cumsum` kernel. The retained infrastructure
incident is `results/infrastructure_failure_122739196.md`. The successful
source changed only the equivalent top-mass reduction; it did not change any
scientific input or threshold.

## Findings

| Prespecified quantity | Observed |
|---|---:|
| Median mass of largest 20% ordered pairs | `0.428908` |
| Mean mass of largest 20% ordered pairs | `0.434069` |
| Molecules reaching 0.80 mass on that subset | `1 / 50,000` |
| Median effective pair support, `exp(entropy)/n²` | `0.777508` |
| Mean mass on true directed bonds | `0.163251` |
| Mean mass on two-hop non-bonded pairs | `0.221197` |
| Mean mass outside self/bond/two-hop categories | `0.550096` |
| Median top-20% mass, >12 atoms | `0.434085` |
| Median top-20% mass, hardest K1-error decile | `0.415215` |

All four prospective concentration gates failed, by a large margin. PairToken
does use a nonuniform learned assignment—20% of pairs carry about 43% of mass,
not the 20% of a uniform distribution—but its useful representation is broad,
especially in K1-hard molecules. The observed K1-to-PairToken development gain
was reproduced at `0.0030436 eV`; this reused selection role is not an
independent transfer result.

## Scientific decision

The proposed *dynamic top-pair truncation* question was rejected before any
new architecture training. A hard top-20% mask would discard a median 57% of
the trained attention mass; restricting interactions to bonds or two-hop pairs
would also discard the observed 55% mean mass outside those categories
(including any disconnected atom pairs). This does not
prove that all efficient relation approximations fail, but it falsifies this
specific concentration premise. No seed, candidate job, scale bridge, official
role, or desktop handoff was released. This diagnostic is `NO_TRAIN` evidence,
not a new replay-ready model candidate; prior candidate trajectories and the
frozen reference remain unchanged.

The literature motivation was relation propagation among nodes and edges in
[GPTrans](https://www.ijcai.org/proceedings/2023/396), pair interactions in
[TGT](https://arxiv.org/abs/2402.04538), and masked/global edge attention in
[ESA](https://doi.org/10.1038/s41467-025-60252-z). Those papers motivate
examining information flow, not assuming that this trained PairToken is sparse.
The direct frozen-checkpoint measurement controls the local decision.
