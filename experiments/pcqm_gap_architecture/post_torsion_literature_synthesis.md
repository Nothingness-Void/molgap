# Post-Torsion Literature Synthesis

Synthesis date: 2026-09-01

## Evidence base

This synthesis re-ranks the 50 primary papers in
[`recent_literature_coverage_ledger_50.md`](recent_literature_coverage_ledger_50.md)
after the accepted sparse-torsion result. Fifteen papers were previously read
to configuration depth and 35 to mechanism/ablation depth. Published scores
remain context only; local advancement still requires the frozen PCQM
100K/10K matched contract.

## Why sparse torsion failed

The candidate completed all 40 epochs, so its failure was scientific rather
than an unfinished optimization or infrastructure result. Four effects explain
the observed combination of slightly worse accuracy and much lower throughput:

1. **Low information increment.** The accepted model already injects every
   ETKDG bond distance and bonded angle into persistent edge and wedge states.
   A single-conformer dihedral is therefore partly determined by information
   already present and adds less independent signal than it would in an
   atom-only model.
2. **Noisy geometric source.** ETKDGv3+MMFF94s is deterministic and consistent,
   but it is not the B3LYP equilibrium geometry. Dihedrals are more sensitive
   than bond lengths and local angles to conformer choice. A narrow persistent
   state can preserve that conformer-specific error through all nine blocks.
3. **Path-count cost rather than parameter cost.** The candidate added only a
   small parameter fraction, but enumerated and updated millions of directed
   four-atom paths. Sparse gather/reduce work, not model size, caused the large
   throughput loss.
4. **Uniform return path.** Every torsion returned context to its three bonds
   and two wedges through mean aggregation. The model had no sparse attention
   mechanism for selecting the few torsions most relevant to conjugation or
   frontier-orbital coupling, so weak/noisy paths could dilute useful ones.

The final epoch was worse than the saved best epoch, and the best checkpoint
still missed the comparator. A longer run is therefore not justified under the
budgeted-convergence contract. This result closes only the fixed width-16
persistent torsion mechanism; it does not negate the three-seed success of
distance-plus-angle bottom fusion.

## What all 50 papers imply after this result

| Literature family | Papers represented in the ledger | Local conclusion |
|---|---|---|
| Persistent atom/bond co-processing | DeMol, Dual Graph Transformer, Edge-Set Attention, MoleculeFormer | Strongest remaining random-initialized route. Their common gain is a separately normalized bond stream and atom--bond exchange, not a larger atom Transformer. |
| Explicit angles/torsions/four-body state | TetraGT, four-body Hybrid Transformer Graph, Fractional/Sliced Denoising | Direct torsion state is now closed locally. The denoising variants remain teacher/pretraining questions; they do not justify a width or seed retry. |
| Ring/fragment hierarchy | RingFormer, Fragment-Biases, HimNet, UMSGFNet, MOL-Mamba | Deterministic ring systems remain a clean later chemistry prior. Learned motif vocabularies and fingerprint fusion change too many inputs. |
| Scalar/vector or equivariant geometry | GeoMFormer, GotenNet, EquiformerV2, SO3krates, ViSNet, E2Former, FreeCG, HotPP, TACE | Potentially useful after bond exchange, but published wins usually require longer training, richer geometry, or force supervision. Only a width-16 order-1 pilot is budget-compatible. |
| Compact body-order bases | CACE, MACE-OFF23 | CACE-like invariants remain a later isolated route. MACE is more credible as a conformer teacher than as a random-init Gap encoder. |
| Geometry/pretraining teachers | 3D-denoiser teacher, SCAGE, Uni-Mol2, EPT, M2UMol, Stereoelectronic Graphs, MIST, UMA, MMFRL, UniGEM, Fractional/Sliced Denoising, MACE-OFF23, AIMNet2 | High potential for a later teacher/input study, but none can establish an architecture-only gain in the present screen. |
| Generic global mixers and expressivity | Graph-Mamba variants, Polynormer, TGT, Edge Transformer, TopNets, Molecular Set Representation | Low information gain here: nine GPS blocks already provide global atom interaction, while dense pair/triplet routes were locally slow and weak. |
| Optimization/function replacements | Strong-GINE reassessment, KA-GNN, Orb/Orb-v3, SevenNet, eSEN | Warn against confusing training horizon or symmetry cost with architecture. Keep the optimizer frozen; test missing information flow first. |
| Property-pattern or language bias | association patterns, functional-group masking | Later pretraining/descriptor questions with leakage or attribution risk; not the next matched screen. |

## Ranked next experiments

| Rank | Mechanism | Information increment | New cache | Expected bounded cost | Decision |
|---:|---|---|---|---|---|
| 1 | Sparse atom--bond dual stream | Selective bond--bond communication plus symmetric atom↔bond exchange | No | One candidate, about 3--3.6 P100 hours | **Submit next** |
| 2 | Ring/conjugation hierarchy | Deterministic chemistry level absent from atom/bond categories | Yes | CPU ring cache plus one candidate | Only if dual stream fails |
| 3 | Width-16 invariant/vector repair | Retains orientation lost by scalar distance/angle bases | No new conformer | Higher implementation and A100 risk | Conditional after bond route |
| 4 | Compact CACE-like basis | Polynomial body-order invariants without explicit torsion paths | Likely yes | Unknown until preflight | Behind ring and vector screens |
| 5 | MACE/AIMNet2/SliDe teacher | Better geometry or transferred geometric knowledge | Separate teacher assets | Not an architecture claim | Separate research track |

## Selected mechanism

The selected experiment retains the accepted atom GPS, EdgeState, sparse
wedge state, and distance-plus-angle geometry. Four interleaved updates give
the 64-dimensional real-bond stream its own four-head segmented attention over
adjacent real bonds. A shared low-rank gated exchange maps endpoint atom
context into bonds and aggregated bond context back into atoms. All new output
projections are zero-initialized, so the candidate begins as the accepted
comparator and changes only the missing communication path.

This choice directly follows the strongest cross-paper consensus and also
addresses the torsion failure's uniform aggregation problem: adjacent bond
messages are selected by attention rather than averaged indiscriminately.
