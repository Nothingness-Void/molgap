# Molecular specialist feasibility decision

## Decision

Do not proceed to a learned Molecular Router from the current 100K evidence.
The label Oracle has very large capacity, but the identity of the winning model
is not sufficiently predictable from inference-visible molecular structure or
model disagreement. The frozen verdict is
`NO_GO_STRUCTURE_NOT_LEARNABLE`.

This is a Phase A-C, no-training diagnostic. It does not reject ensembling and
does not claim that no future specialist can ever be routable. It rejects
building a Router from this reused development role and the current expert
pool without new evidence.

## Phase A: asset audit

The audit found 59 copies of 39 unique, exactly row-aligned prediction vectors
on the accepted 50,000-row PCQM development range `[100000,150000)`:

- 33 K1-lineage V4 endpoints form the primary Oracle pool;
- 5 GPTrans endpoints share rows and targets but use the separate GPTrans
  optimizer/schedule contract, so they are contextual only;
- PairToken+MoSE shares rows and targets but has the already-recorded
  undeclared feature-identity mismatch, so it is contextual only.

Every included payload has exactly the same `source_idx` and float target
tensor as immutable K1-v4. Source CSV indices and Gap values were independently
cross-checked. Official validation, test-dev and test-challenge were not read.
The aligned row-level matrix is retained outside Git at
`platforms/_records/local/pcqm_molecular_router_audit/aligned_prediction_matrix.pt`
(SHA-256
`c52e944def35d8d9ccb41011c9cf820182702c1a9006189b6fc488397169edbb`).

Older GPS7/GPS9/GPS11, dense/equal and PubChemQC assets cannot enter this
matrix: they use a different corpus and three targets. QM9 outputs use another
dataset, older PCQM screens use a different 10K development split, and 500K
endpoints use source rows `[500000,550000)`. No regenerated predictions are
needed for this 100K question. A separate 500K routing question would require
the matched K1 reference that belongs to the active scale-attribution chain.

Full paths, payload hashes, aliases and compatibility classes are in
`results/asset_audit.json`.

## Phase B: Oracle capacity

K1-v4 MAE is `0.1413736414 eV` on this reused development role.

The strongest primary-pool pairwise Oracle uses the globally worse
`neural_atom_k1_mose_hidden_bn` endpoint:

| Quantity | Value |
|---|---:|
| Expert MAE | `0.1426464517 eV` |
| Oracle(K1, expert) MAE | `0.1072343137 eV` |
| Oracle gain | `0.0341393277 eV` |
| Expert win rate | `49.608%` |
| Mean winning margin | `0.0688181900 eV` |
| Minimum Oracle gain across five deterministic folds | `0.033845 eV` |

Original PairToken gives a similar result: expert MAE `0.1383300447 eV`,
pairwise Oracle MAE `0.1080273338 eV`, Oracle gain `0.0333463076 eV`, and win
rate `51.244%`. Thus the observed behavior is not a rare 5% specialist pocket;
about half the rows switch winner.

The greedy primary-pool Oracle curve is:

| Experts including K1 | Oracle MAE | Gain vs K1 |
|---:|---:|---:|
| 1 | `0.1413736414` | `0` |
| 2 | `0.1072343137` | `0.0341393277` |
| 3 | `0.0907933277` | `0.0505803137` |
| 5 | `0.0749101163` | `0.0664635250` |
| 10 | `0.0585769388` | `0.0827967025` |
| 33 | `0.0415071483` | `0.0998664931` |

This decisively passes the frozen Oracle-capacity gate. It is nevertheless a
label-selected, non-deployable ceiling and is inflated by accumulating many
correlated single-seed endpoints on a role already used for model selection.

As a non-Router diagnostic, fixed 0.5/0.5 averaging of K1 and original
PairToken reaches `0.1340852096 eV`, a `0.0072884317 eV` gain over K1. This was
identified post hoc on the reused role and is not a promotion, but it suggests
variance cancellation is more immediately plausible than discrete molecular
specialization.

Exact pairwise distributions and the full curves are in
`results/oracle_report.json`, `results/pairwise_oracle.csv`, and
`results/primary_greedy_curve.csv`.

## Phase C: structural regularity

The five primary experts with the largest pairwise Oracle gains were tested
against RDKit descriptors, graph/RWSE descriptors, absolute prediction
disagreement, all-primary prediction variance, Bemis-Murcko scaffolds, and 64
unsupervised Morgan-fingerprint clusters.

The strongest scalar average-precision lift over the expert-win base rate was
only `0.01176`; PairToken's absolute-disagreement lift was `0.00884`. Both are
far below the frozen `0.05` requirement. Descriptor-quintile win-rate spreads
were small, and no Morgan cluster simultaneously had at least 500 rows,
`0.10` win-rate uplift, and positive uplift in all five folds. Some tiny
20--28-row scaffolds look enriched, but they are descriptive, unstable under
multiple inspection, and cannot satisfy the gate.

One official SMILES with unusual silicon valence was rejected by the installed
RDKit sanitiser. It remained in all 50,000-row Oracle calculations and was
excluded only from RDKit/scaffold/Morgan analyses; graph-native descriptors
remained available. No row or prediction was dropped from Phase A-B.

Aligned frozen embeddings, seed variance and a train-reference OOD-distance
asset were not available, so no claims are made for them. Their absence does
not explain the failure of the already-frozen visible-feature gate.

## Interpretation

The current expert pool contains substantial residual diversity, but the
winner boundary behaves much more like training/model noise than a stable
chemical partition. A Router trained now would have a large Oracle to chase
and very little observable signal telling it when to switch; false routes can
therefore erase the apparent headroom.

Phase D is not authorized. If this direction is revisited, it needs a new,
prospectively frozen role or genuinely aligned uncertainty/embedding evidence,
not a more complex Router on this development set. Fixed averaging may be
posed as a separate future question, but this audit does not release it.

