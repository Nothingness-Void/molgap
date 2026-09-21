# K1 SPD-PairToken decision

Kaggle3 run `nvoid912/molgap-k1-spd-pairtoken-s42:v1` completed all 40
epochs and passed saved-artifact acceptance without model inference. It used
the fixed PCQM-100K V4 contract, direct Gap, seed 42, strict FP32/no TF32,
physical batch 128, and no protected role.

| Model | Parameters | Development MAE | Difference vs K1 |
|---|---:|---:|---:|
| Frozen K1-v4 | 3,658,817 | `0.1413736343 eV` | — |
| Original PairToken | 3,681,665 | `0.1383300447 eV` | `-0.0030435896 eV` |
| K1 SPD-PairToken | 3,681,672 | `0.1397010982 eV` | `-0.0016725361 eV` |

The candidate-minus-K1 paired absolute-error interval was
`[-0.0025584301, -0.0007845981] eV`, so the direction is reliably favorable.
The gain nevertheless missed the prospectively frozen `0.003 eV` material and
run-variation gate. More importantly, the candidate was worse than its actual
parent, original PairToken, by `0.0013710489 eV`; that paired interval was
entirely unfavorable at `[+0.0005010213, +0.0022163865] eV`.

The learned seven shortest-path biases were non-trivial:
`[+0.2574, -0.0649, -0.3200, -0.5008, -0.1033, +0.0688, +0.3524]` for
distance buckets `0, 1, 2, 3, 4, 5, >=6`. The mechanism therefore trained;
this is not an inactive-path failure. It favored self-pairs and the longest
bucket while suppressing intermediate distances.

Relative to K1, the candidate damaged the easiest three K1-error quintiles by
`+0.04018`, `+0.01895`, and `+0.00012 eV`, while improving the hardest two by
`-0.01852` and `-0.04910 eV`. This repeats the allocation pattern seen in the
full-depth sparse-triplet experiment: explicit topology helps difficult rows
but perturbs easy rows. Relative to original PairToken, SPD was worse in four
of five K1-error quintiles. A single globally shared distance prior is thus a
poor way to decide when topological conditioning should apply.

Runtime was effectively unchanged from K1 (`809.51` versus `812.30 graphs/s`),
and memory reserve remained above 95%, so the negative decision is scientific,
not a resource failure. SPD-conditioned PairToken is retained as
`POSITIVE_BELOW_GATE` evidence against K1 but is not promoted; shortest-path
bucket, cutoff, seed, or width retries are closed. One final authorized round
may test a distinct allocation mechanism, but it must not add another path
value stream, persistent triplet state, or globally shared topology bias.
