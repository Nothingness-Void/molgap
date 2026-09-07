# Incremental Literature Extension for PCQM Gap (2026-09-07)

This document extends the base literature ledger without renumbering or
rewriting it. The base ledger remains the authority for its 73 audited
sources:
[`recent_literature_coverage_ledger_50.md`](recent_literature_coverage_ledger_50.md).
This supplement records additional sources found after that ledger and maps
them to the current Track B constraints. It is research evidence only: it does
not change `CURRENT_STATE.md`, `ROADMAP.md`, the active K3 question, or any
remote-run authorization.
For the separate official-board versus paper-validation audit, see
[`leaderboard_sota_audit_2026-09-07.md`](leaderboard_sota_audit_2026-09-07.md).

The selection rule is deliberately narrow. A source is retained only when it
adds at least one of the following: (i) a distinct information channel that
could fit the bounded PCQM screen, (ii) a credible electronic-structure
teacher or auxiliary target, (iii) a geometry/data route that can be tested
under an explicit new input contract, or (iv) a methodological warning about
benchmark comparability. Reported numbers from a different split, target,
level of theory, or inference contract are not treated as MolGap results.

## Project anchor and the new conclusion

The project is selecting a Gap-only PCQM4Mv2 specialist. The current accepted
anchor is the no-attention shared GraphState model, confirmed against fresh
full-GPS controls at seeds 42/43/44. The active server-side question is K3b:
persistent conjugated-component communication against the exact
descriptor-only control. The architecture screen is random-initialized, uses
the official-train-derived 100K/10K split, keeps official validation/test-dev
sealed, and requires ETKDG-consistent geometry whenever geometry is used.

The additional reading changes the *shape* of the remaining search rather than
adding a long list of direct candidates:

1. One bounded direct architecture remains defensible: a static hop/path-count
   channel in the style of MetaGIN. It is 2D path information, not a new noisy
   torsion coordinate state, and should be screened only after the active K3b
   result.
2. The stronger recent signal is electron-aware supervision. HEDMoL, MET,
   atom-level quantum pretraining, Q-GEM, QCDGE, and bond-level overlap
   population prediction all add electronic information through teachers,
   auxiliary labels, or pretraining. None is evidence for another
   random-initialized architecture branch under the current contract.
3. Geometry remains a conditional route, not a default. GeoOpt-Net,
   nablaColors-3D, QO2Mol, and VQM24 make the geometry question more precise,
   but they use different levels of theory, targets, or inference conformers.
4. The literature does not justify a full-GPS expansion, another generic
   contact-edge screen, or a second torsion hierarchy. The remaining value is
   in changing the information source and scientific contract, not in stacking
   another large block onto the accepted encoder.

## A. Direct, bounded architecture evidence

### E1. MetaGIN: static hop/path information

**Source.** [MetaGIN: a lightweight framework for molecular property
prediction](https://academic.hep.com.cn/fcs/EN/10.1007/s11704-024-3784-y),
Frontiers of Computer Science 2025.

**What was actually useful.** MetaGIN reports a PCQM4Mv2 MAE of `0.0851` with
`8.87M` parameters. Its central mechanism is a MetaFormer-style stack with a
3-hop convolution mixer and a graph-propagation mixer. The edge features are
explicitly separated by hop: 1-hop bond type/chirality/conjugation/rotatable
flags, 2-hop path counts, and 3-hop path counts. Its ablation moves from
`0.0920` at 1 hop and `2.82M` parameters, through `0.0881` at 2 hops and
`3.63M`, to `0.0871` at 3 hops and `4.45M`. Increasing width to 512 gives
`0.0851` at `8.87M`.

**Transfer judgment.** This is the only newly found source that suggests a
small, graph-only information channel not already represented by the accepted
GraphState/RWSE path: deterministic hop/path-count statistics.
The paper's statement that 3-hop features represent 3D structure should not be
accepted literally; the reported PCQM input is still a 2D graph with static
path features. The result is therefore a hypothesis about long-range topology
and conjugation, not evidence that a new torsion coordinate state will help.

**Bounded candidate.** If K3b does not clear the material-gain gate and the
discovery loop remains open, test only a width-reduced, `<=4M` variant:
deterministic 2-hop/3-hop path-count features or a sparse hop-state update,
with the existing GraphState anchor and no pretraining, no new geometry, and
no change to the split or target. The 4.45M ablation is above the project's
parameter ceiling and cannot be copied as-is. The paper does not expose enough
matched seed/split/budget detail to accept its PCQM number directly.

### E2. Attention-based functional-group coarse-graining

**Source.** [Attention-based functional-group coarse-graining: a deep learning
framework for molecular prediction and design](https://www.nature.com/articles/s41524-025-01836-7),
npj Computational Materials 2025.

**What was actually useful.** The model has an atom graph and a deterministic
functional-group/motif graph. Atomic embeddings are pooled into functional
group embeddings, a second message-passing network processes the motif graph,
and an attention-based property head aggregates group context. Ring systems
are kept as self-contained motifs so aromaticity and conjugation are not split
apart. On QM9, the paper reports approximately `R^2=0.97` for HOMO and LUMO
using only about 6,000 molecules, alongside high scores on other QM9 targets.

**Transfer judgment.** The result is coupled to a variational graph
autoencoder, property-guided generation, and multi-task training on a small
QM9 setting. It is not a matched PCQM Gap architecture comparison. Its useful
lesson is narrower: a deterministic conjugated/ring component can be a
separate representation level, but the already closed Ring-GraphState screen
means that another atom-to-ring message-passing branch is not automatically
new evidence. The electronic teacher route is more novel than another ring
token route.

## B. Electron-aware representation and auxiliary supervision

### E3. HEDMoL: electron-informed substructure transfer

**Source.** [Electron-Informed Coarse-Graining Molecular Representation
Learning for Real-World Molecular Physics](https://arxiv.org/html/2602.07087),
KDD 2025, DOI `10.1145/3690624.3709270`.

**Mechanism.** HEDMoL decomposes a molecule with a junction-tree procedure,
retrieves electron-level attributes for each substructure from an external
QM9 database using graph similarity, builds an electron-derived substructure
graph, and jointly learns atom-level and electron-level embeddings. It adds a
physical-consistency regularizer that makes the atom and retrieved-electron
views agree on a potential-energy quantity. The ablation shows that the
knowledge-extension step provides the main jump over the base GNNs, while
hierarchical learning and the consistency regularizer provide smaller but
occasionally useful additions. The authors explicitly do not regard direct
QM9 evaluation as fair because QM9 is also used as the external source of
retrieved quantum attributes.

**MolGap implication.** This is a strong design for a *teacher or auxiliary
descriptor* experiment: deterministic conjugated components can be given
charge/orbital-like attributes from a disjoint small-molecule database, then
the target model can consume a frozen or jointly trained electron-aware
representation. It is not a free architecture gain. Any MolGap adaptation
must deduplicate by molecular identity, make the external source explicit,
and report whether the source contains PubChemQC/PCQM molecules. HEDMoL also
roughly doubles the embedding computation because it runs two GNN paths; the
current 12-hour screen should not copy the complete model.

### E4. MET: charge-supervised equivariant pretraining

**Source.** [Integrating equivariant architectures and charge supervision for
data-efficient molecular property prediction](https://doi.org/10.1039/D5ME00173K),
Molecular Systems Design & Engineering 2026.

**Mechanism.** The Molecular Equivariant Transformer (MET) combines an EGNN
encoder with Transformer layers and pretrains on quantum-derived atomic
partial charges. The paper reports improvements across molecular property
tasks, especially in low-data regimes, and analyzes functional-group patterns
and alignment with molecular dipoles. Its ablations attribute transferable
spatial features primarily to the EGNN encoder and task adaptation to the
Transformer layers.

**MolGap implication.** The transferable idea is charge supervision, not a
new EGNN/Transformer transplant. A later post-selection experiment could
pretrain a compact 2D/3D teacher on atomic charges and transfer its frozen
features to Gap. The current random-init architecture screen cannot include
that pretraining and still claim an architecture-only comparison.

### E5. Atom-in-a-molecule quantum-property pretraining for Graphormer

**Source.** [Pretraining graph transformers with atom-in-a-molecule quantum
properties for improved ADMET modeling](https://link.springer.com/article/10.1186/s13321-025-00970-0),
Journal of Cheminformatics 2025.

**What was actually useful.** The study compares Graphormer pretraining on
atom-resolved quantum properties, molecular HOMO-LUMO gaps, and atom masking.
Across its public ADMET evaluation, atom-level quantum pretraining and masking
generally outperform gap pretraining; on a larger internal clearance task,
atomic charges and NMR shifts are the most useful choices. Its attention-rollout
analysis also connects the learned representation to low-frequency graph
spectral behavior.

**MolGap implication.** This is the clearest evidence in the new set against
using the same molecular Gap as the only teacher target. If a teacher budget is
available, start with local electronic labels—charges, bond orders, or
orbital/bond descriptors—and test readout-only transfer before building a
large pretrained backbone. The source is downstream ADMET evidence, so it does
not establish a PCQM Gap gain.

### E6. Q-GEM: geometry plus electronic SSL

**Source.** [Q-GEM: Quantum Chemistry Knowledge Fusion Geometry-Enhanced
Molecular Representation for Property Prediction](https://doi.org/10.1002/advs.202504867),
Advanced Science 2025.

**Mechanism.** E-GeoGNN passes information among an atom-bond graph, a
bond-angle graph, and an angle-dihedral graph. Its first pretraining stage
uses about 20 million Zinc20 molecules with MMFF94 geometries to predict bond
lengths, angles, dihedrals, and pair distances. Its second stage uses the
154,610-molecule QuanDB dataset to predict CM5 charges and Wiberg bond orders
from stable DFT conformers.

**Important negative evidence.** The paper reports only small gains from the
electronic SSL stage on the quantitative QM tasks (`0.6%` on QM8 and `0.2%` on
QM9 in the cited comparison), and attributes this partly to theoretical-level
and chemical-space mismatch. It also explicitly omits long-range and
intermolecular interactions.

**MolGap implication.** Q-GEM should not be used to reopen the failed local
torsion-state route. Its valuable lesson is that a geometry/electronic teacher
needs label-level alignment; a large mismatched charge database can be less
useful than a smaller target-level source. It is a reference for a later
teacher contract, not a current direct candidate.

### E7. Bond-level overlap population prediction

**Source.** [Prediction of the overlap population diagram of organic molecules
based on a graph neural network](https://doi.org/10.1093/chemle/upaf038),
Chemistry Letters 2025.

**What was actually useful.** The authors construct an approximately
`2.37M`-bond database over `130,000` QM9 molecules and train a GNN to predict
bond overlap-population diagrams. The calculations use FHI-aims with PBE;
the model uses 128 hidden channels, a 256-dimensional output, four message
passing blocks, radial/spherical basis sizes 6/7, a 5 Å cutoff, Adam at
`1e-4`, and a 6:2:2 molecule split. The model reproduces bonding/antibonding
trends and substituent effects qualitatively, but the paper also shows
quantitative errors for strained or unfamiliar structures.

**MolGap implication.** OP is a plausible bond-level auxiliary target because
it is closer to orbital interaction than an arbitrary contact edge. It is
still PBE/QM9 data rather than B3LYP/6-31G* PCQM data, so it belongs in a
teacher or auxiliary-label pilot with a strict no-overlap audit. It should not
be injected into the random-init screen as if it were an input available at
inference.

### E8. AEGCNN-MTL: related-target sharing and negative transfer

**Source.** [Adaptive edge-aware graph convolutional with multi-task learning
for simultaneous prediction of material properties](https://www.nature.com/articles/s41524-025-01917-7),
npj Computational Materials 2026.

**What was actually useful.** On its QM9 evaluation, the authors group HOMO,
LUMO, and Gap as a correlated frontier-orbital group. On their DFT band-gap
and work-function dataset, the multi-task model improves the band-gap MAE
relative to its single-task version; on weakly correlated dipole/spatial-extent/
ZPVE groups, joint training is less stable and the authors use single-task
models. The architecture is a shared encoder with Set2Set pooling,
edge-aware convolution, distance-modulated attention, and task-specific
feed-forward heads.

**MolGap implication.** This supports a post-selection experiment that predicts
HOMO, LUMO, and Gap jointly *only if* those labels are available under the same
data and split contract. It does not authorize changing the current Track B
Gap-only screen. The negative-transfer result is as important as the positive
one: auxiliary targets need measured correlation and loss balancing, not just
more heads.

## C. New quantum-chemistry data sources

### E9. QCDGE: near-target-level ground and excited states

**Source.** [QCDGE database, Quantum Chemistry Database with Ground- and
Excited-state Properties of 450 Kilo Molecules](https://arxiv.org/html/2406.02341),
arXiv 2024 preprint.

**What was actually useful.** QCDGE retains `443,106` small molecules with up
to ten heavy atoms from C/N/O/F. Ground-state geometry optimization and
frequency calculations use B3LYP/6-31G* with BJD3; the database includes
ground-state HOMO/LUMO, Mulliken charges, geometry, thermal and vibrational
properties, plus TD-DFT excited-state information. The source records include
QM9, PubChemQC and GDB-derived molecules, and the authors provide HDF5 data
with SHA-512 integrity hashes.

**MolGap implication.** This is the strongest new candidate for a target-level
teacher because its ground-state method is close to the project's B3LYP/
6-31G* target. It is not automatically clean: the BJD3 correction, optimized
geometry contract, and PubChemQC source overlap must be audited. A safe pilot
would deduplicate QCDGE against PCQM by canonical identity, keep QCDGE labels
out of the target split, and test charge/Mulliken/HOMO/LUMO auxiliary transfer
with a separate frozen protocol.

### E10. QuanDB: broad local electronic labels

**Source.** [QuanDB: a quantum chemical property database towards enhancing 3D
molecular representation learning](https://link.springer.com/article/10.1186/s13321-024-00843-y),
Journal of Cheminformatics 2024.

QuanDB contains `154,610` molecules, nine elements, 53 global and 5 local
quantum-chemical properties, and a lowest-energy conformation. Geometry and
single-point calculations use solvated B3LYP-D3(BJ) levels with
6-311G(d)/def2-TZVP choices, and the reported total computational cost exceeds
`10^7` core-hours. Local properties include multiple charge definitions and a
bond-order quantity.

**MolGap implication.** QuanDB is a strong source for charge/Wiberg-style
teacher pretraining and a weak source for direct target calibration: the
solvent, dispersion, basis, and geometry differ from PCQM. The paper should be
used together with Q-GEM, not counted as independent evidence for a direct Gap
architecture.

### E11. QO2Mol: many conformers at B3LYP/def2-SVP

**Source.** [An Open Quantum Chemistry Property Database of 120 Kilo Molecules
with 20 Million Conformers](https://arxiv.org/abs/2410.19316), arXiv 2024.

QO2Mol contains about `120,000` organic molecules, approximately `20M`
conformers, ten elements, and molecules extending beyond 40 heavy atoms. The
conformers are computed at B3LYP/def2-SVP and provide energies, forces, formal
charge and related attributes. Molecules are assembled from ChEMBL fragments,
so the distribution is more drug-like than QM9.

**MolGap implication.** This is a geometry/PES pretraining source, not a
drop-in Gap label source. It can support a conformer teacher or robustness
study for larger molecules, but the basis, property set, and many-conformer
sampling contract must remain explicit. It cannot be mixed with ETKDG
inference while retaining the current train/inference claim.

### E12. VQM24: exhaustive small-molecule quantum coverage

**Source.** [Quantum mechanical dataset of 836k neutral closed-shell molecules
with up to 5 heavy atoms](https://www.nature.com/articles/s41597-025-05428-4),
Scientific Data 2025.

VQM24 enumerates `836k` neutral closed-shell molecules over nine elements and
provides DFT properties plus diffusion quantum Monte Carlo values for a
subset. It covers the small constitutional-isomer space more exhaustively
than QM9 and reports that atomization-energy learning is harder than on QM9.

**MolGap implication.** VQM24 is useful for evaluating whether a representation
is learning chemistry rather than memorizing the QM9 distribution, and for
pretraining compact atom-level teachers. It is not a PCQM replacement: the
size range, element set, charge/spin restriction, geometry, and level of
theory differ. Use only in a separate OOD/teacher contract.

## D. Geometry fidelity, fragments, and conjugation-specific transfer

### E13. GeoOpt-Net: single-step geometry refinement

**Source.** [A Cross-Domain Graph Learning Protocol for Single-Step Molecular
Geometry Refinement](https://arxiv.org/html/2601.22723), arXiv 2026;
the manuscript identifies DOI `10.1021/acs.jctc.6c01080`.

**Mechanism.** GeoOpt-Net starts from an RDKit ETKDGv3 + MMFF94 conformer and
uses three SE(3)-equivariant streams for bond lengths, angles, and dihedrals,
followed by a lightweight Transformer decoder. It pretrains on QM9/QM40 at
B3LYP/6-31G(2df,p), fine-tunes on QMe14S at B3LYP/TZVP, and uses
fidelity-aware feature modulation to calibrate the theory level. The external
ZINC20 evaluation claims sub-milliångström geometry error and improved DFT
convergence.

**MolGap implication.** This gives the project a precise future geometry
hypothesis: improve the input conformer before asking the Gap encoder to learn
electronic effects. It changes the conformer method, however, and its target
level is B3LYP/TZVP rather than the project's B3LYP/6-31G* target. The extreme
single-example numbers should be treated as manuscript claims until code and
independent reproduction are available. Any adoption requires a new immutable
geometry cache and the same geometry source in training and inference.

### E14. nablaColors-3D: conformer fidelity and solvent-aware evaluation

**Source.** [A conformational benchmark for optical property prediction with
solvent-aware graph neural networks](https://www.nature.com/articles/s42004-026-01944-5),
Communications Chemistry 2026.

The paper curates `26,369` chromophore-solvent pairs with multiple conformers,
uses scaffold splits, and compares xTB, vacuum-DFT, and implicit-solvent-DFT
geometry sources. It pretrains 3D backbones on PCQM4Mv2 and reports that
higher-fidelity conformers often help, while a UniMol+-derived model can reduce
the sensitivity to inference geometry by learning a refinement mapping. The
paper also shows that random solvent-pair splitting can leak the same
chromophore across train and test.

**MolGap implication.** This is an evaluation-design source more than an
architecture source. It supports measuring conformer sensitivity while holding
the encoder fixed, and it reinforces the project's ETKDG train/inference
consistency rule. It does not prove that solvent or DFT conformers improve
PCQM Gap under the current contract.

### E15. FACET: fragment-aware multi-conformer aggregation

**Source.** [FACET: A Structure-Aware Graph Transformer for Molecular Property
Prediction](https://proceedings.iclr.cc/paper_files/paper/2026/hash/74a43cca51015f8fbf91f0154f13bfa1-Abstract-Conference.html),
ICLR 2026.

FACET combines a fragment-level 2D graph with multiple 3D conformer
representations and learns a Transformer proxy for fused Gromov-Wasserstein
alignment. Its RingPath extraction treats rings, acyclic paths, and junctions
as deterministic fragments; the paper reports scaling to large molecular
collections and many conformers with a large speed advantage over direct FGW
baselines. The results are on MoleculeNet/MARCEL rather than PCQM HOMO/LUMO.

**MolGap implication.** The portable idea is not the full alignment objective:
it is a fragment cache plus a multi-conformer teacher whose final inference
can remain compact. This can be revisited only after the direct architecture
selection, and only with an explicit conformer-ensemble contract. It should
not be confused with the already closed single-ring GraphState screen.

### E16. Domain-matched pretraining for conjugated polymers

**Source.** [Fine-Tuning Directional Message Passing Neural Networks:
Predicting Properties of Conjugated Organic Polymers with High
Accuracy](https://pubmed.ncbi.nlm.nih.gov/41977627/), Polymers 2026,
DOI `10.3390/polym18070879`.

The study uses a DimeNet++-style directional message-passing model on 3D
monomer structures to predict polymer HOMO, LUMO, and gap. Pretraining on
TD-DFT-extrapolated polymer-relevant data produces much lower reported errors
than direct training, while pretraining on monomer DFT data does not provide
comparable gains. The abstract reports approximate final MAEs of `0.074 eV`
for HOMO, `0.141 eV` for LUMO, and `0.172 eV` for the gap in its own polymer
setting.

**MolGap implication.** This is strong evidence for *domain-matched* teacher
data in conjugated systems, not for another DimeNet++ transplant. If the K3
component route is scientifically positive, the next question should be
whether a conjugation-relevant auxiliary corpus helps; generic monomer/QM9
pretraining is not enough by assumption.

## E. Long-range and higher-fidelity targets

### E17. CELLI: charge equilibration for non-local interactions

**Source.** [Learning non-local molecular interactions via equivariant local
representations and charge equilibration](https://www.nature.com/articles/s41524-025-01790-4),
npj Computational Materials 2025.

CELLI inserts a differentiable charge-equilibration layer into equivariant
machine-learned potentials. It splits the energy into a Coulomb term and a
short-range correction, globally equilibrates partial charges, and conditions
the local correction on the resulting charge environment. The reported
experiments on Allegro, MACE, OE62 and SPICE target charge transfer and
long-range interactions.

**MolGap implication.** CELLI is a principled alternative to arbitrary
through-space contact edges, but it solves an energy/force problem with charge
state and geometry inputs. It is not evidence for a direct PCQM Gap branch.
If contact-edge failure is followed by a future long-range experiment, a
charge-equilibration/teacher contract is more scientifically distinct than
another cutoff or edge-width variant.

### E18. Transfer learning to GW/BSE excitation energies

**Source.** [Transfer learning of GW Bethe-Salpeter equation excitation
energies](https://pubs.rsc.org/en/content/articlehtml/2026/sc/d5sc09780k),
Chemical Science 2026.

The study pretrains a graph model on large DFT/TDDFT corpora and fine-tunes on
small, expensive quasiparticle GW and GW-BSE data. It compares readout-only
adaptation with full-model fine-tuning and shows that low-fidelity pretraining
can reduce the expensive-label requirement while retaining accuracy on larger
or chemically distinct test molecules.

**MolGap implication.** This supports the project's separate high-fidelity
Delta-GW direction and the principle that readout-only transfer should be
tested before full fine-tuning. It does not alter the current B3LYP Gap target,
and no GW/BSE quantity should be mixed into the Track B architecture screen.

### E19. Systematic review of frontier-orbital prediction

**Source.** [Machine learning for frontier orbital energetics: A review of
HOMO-LUMO prediction methods](https://doi.org/10.1016/j.cartre.2026.100674),
Carbon Trends 2026.

The review applies a reproducible search and retains 59 studies from 2022 to
early 2026, extracting architecture, physical prior, benchmark, cost,
generalization and uncertainty fields. Across the reviewed corpus, the median
reported MAE is lower for equivariant than invariant models (`25.8` versus
`52.0 meV`), but the distributions overlap and are confounded by model size,
pretraining and benchmark differences. The review also finds that complexity,
latency and uncertainty are frequently unreported, and it stresses the
limitations of QM9's small, neutral, closed-shell, single-conformer regime.

**MolGap implication.** The actionable result is methodological: the field is
not short of architectures; it is short of matched cost, split, uncertainty,
and generalization reporting. This supports the project's strict paired gates
and argues against treating leaderboard numbers from other contracts as a
reason to expand the current screen.

## F. Adjacent references for bounded routes already selected for the working tree

These are included to make the current directed-bond and Laplacian-PE
hypotheses auditable. They are not a statement that the referenced routes have
passed a scientific gate or changed the live `CURRENT_STATE.md`.

### E20. SignNet and BasisNet

**Source.** [Sign and Basis Invariant Networks for Spectral Graph
Representation Learning](https://arxiv.org/abs/2202.13013), 2022.

SignNet addresses eigenvector sign flips; BasisNet additionally handles the
more general basis ambiguity in repeated or near-degenerate eigenspaces. A
per-eigenvector `(+v,-v)` construction is evidence for sign invariance only.
If the current LapPE path uses fixed low modes, it must either audit spectral
degeneracy or avoid claiming full basis invariance. This is a correctness gate,
not a reason to add more eigenvectors.

### E21. Directed Graph Attention Networks

**Source.** [Directed message passing based on attention for prediction of
molecular properties](https://www.sciencedirect.com/science/article/pii/S0927025623004378),
Computational Materials Science 2023.

D-GATs split each molecular bond into directed states and update bond and atom
states with scaled dot-product attention. The authors report gains on many
MoleculeNet tasks and attribute them to better functional-group context.
This supports a bounded directed non-backtracking relation update, but the
current lightweight hypothesis uses mean predecessor aggregation rather than
attention. There is no direct PCQM Gap result, so only the information-flow
idea transfers.

### E22. Deep Positional Encoders

**Source.** [Deep positional encoders for graph
classification](https://www.sciencedirect.com/science/article/abs/pii/S0031320325014918),
Pattern Recognition 2025.

Deep Positional Encoders learn Laplacian eigenvectors through Dirichlet-energy
objectives rather than relying entirely on per-graph eigendecomposition. The
paper evaluates molecular and large graph benchmarks, including a PCQM4Mv2
subset. It is a possible fallback if cached spectral eigenpairs become the
dominant CPU/storage cost, but it changes the cache and pretraining contract;
it is not a reason to replace the current deterministic LapPE cache during the
active screen.

## G. Electronic-density frontier found in the second search

These sources are newer than the fixed 73-source ledger and are especially
important because they move beyond topology toward a representation of the
electronic state itself. They are teacher/data candidates, not permission to
change the active random-init architecture contract.

### E23. EDBench: PCQM-scale electron-density supervision

**Source.** [EDBench: Large-Scale Electron Density Data for Molecular
Modeling](https://arxiv.org/abs/2505.09262), NeurIPS 2025 Datasets and
Benchmarks Track; [official project page](https://hongxinxiang.github.io/projects/EDBench/).

EDBench computes electron densities for `3,359,472` molecules derived from
PCQM4Mv2 with Psi4 B3LYP and element-dependent 6-31G**/6-31+G** basis sets. It
also exposes orbital energies around HOMO/LUMO, energy components, multipole
moments, open-shell labels, cross-modal retrieval, and electron-density
generation tasks. This is the closest newly found large-scale electronic
teacher to MolGap's scientific target, although the basis and data lineage are
not identical to B3LYP/6-31G*.

**Critical gate.** The public repository currently marks the full ED dataset,
property-task code and checkpoints as TODO/release work. More importantly,
EDBench is derived from PCQM and its public task files use their own
scaffold/random splits. Before using it, derive exact CID/canonical-identity
overlap against every MolGap role and exclude any ED/orbital target from
official validation/test molecules. Otherwise a nominal “teacher” would expose
the held-out target or a near-duplicate.

**MolGap implication.** Retain EDBench as a high-priority post-selection
electronic-teacher candidate. The first clean experiment would be a
train-role-only ED or orbital auxiliary task with an identity manifest, exact
level-of-theory record, and a no-teacher control. It must not be used as a
direct replacement for PCQM Gap labels or as an unsupervised pretraining source
until the overlap audit passes.

### E24. ED-DiT: physics-guided electron-density pretraining

**Source.** [ED-DiT: Physics-Guided Diffusion Pretraining for Transferable
Molecular Representations from Electron Density](https://arxiv.org/abs/2608.03260),
August 2026 preprint.

ED-DiT uses masked diffusion denoising over electron density with an
electron-number consistency constraint. On EDBench it improves the same-model
scratch baseline, including orbital-energy RMSE from `0.0293` to `0.0138` with
only `10%` labels, and reports strong gains on electron-density generation and
open-/closed-shell prediction. This is a current frontier pretraining pattern:
the conserved electronic quantity is an explicit constraint rather than a
generic reconstruction loss.

**Transfer judgment.** It is not a PCQM Gap result, and it inherits EDBench's
identity, basis, coordinate, and availability questions. The useful idea is an
auxiliary electron-number/charge-consistency objective after architecture
selection; the full diffusion model is far outside the current Track B screen.

### E25. ELECTRA: floating-orbital charge-density representation

**Source.** [ELECTRA: A Cartesian Network for 3D Charge Density Prediction with
Floating Orbitals](https://proceedings.neurips.cc/paper_files/paper/2025/file/288b63aa98084366c4536ba0574a0f22-Paper-Conference.pdf),
NeurIPS 2025 Spotlight; [code](https://github.com/Jotels/ELECTRA_2025).

ELECTRA represents charge density with learned floating Gaussian orbitals and
predicts their positions, weights and covariances with a Cartesian equivariant
network. The reported downstream value is electron-density reconstruction and
approximately `50.72%` fewer SCF iterations when the predicted density
initializes DFT, not PCQM HOMO/LUMO/Gap prediction.

**Transfer judgment.** Keep it as an electronic representation reference. It
supports a future density/charge teacher but does not justify adding a
Cartesian floating-orbital head to the current 4M random-init architecture.

## H. Leaderboard anomaly and contract-negative controls

### E26. MoiréGT and RadialFocus: apparent 46 meV PCQM results

**Sources.** [MoiréGT](https://openreview.net/pdf?id=sJzfxRbEv6) and
[RadialFocus](https://doi.org/10.1145/3746252.3760877).

These works report approximately `0.0463 eV` PCQM validation error, with
MoiréGT also listing `0.0464 eV` test-dev. The numbers are useful as a search
signal, but the MoiréGT experiment explicitly treats PCQM as a **3D graph**
task and computes distances from physical locations. The official PCQM4Mv2
contract does not provide explicit validation/test coordinates at inference.
RadialFocus presents the same kind of 3D PCQM result without an official
test-dev submission record.

**Disposition.** Retain the papers as legitimate 3D distance-modulated
attention references and as a benchmark-audit negative control. Exclude their
PCQM numbers from the 2D leaderboard and from MolGap candidate selection until
an independent coordinate-free, official-role reproduction is available. A
low number without an audited inference-information statement is not evidence
of a better MolGap route.

## Decision matrix for the remaining search

| Route | New information | Overlap with closed routes | Contract class | Priority |
|---|---|---|---|---:|
| MetaGIN-style path counts | Static 2D 2/3-hop path/conjugation statistics | Low; not a coordinate torsion state | One seed-42 random-init screen, `<=4M`, same 100K/10K split | P1 after K3b |
| QCDGE auxiliary teacher | Near-target B3LYP/6-31G* HOMO/LUMO, Mulliken, excited-state labels | Low, but source overlap risk is high | Frozen external teacher; identity dedup and level-of-theory audit | P1 post-selection |
| HEDMoL/OP/charge teacher | Substructure, bond, or atom electronic descriptors | Distinct from contacts/rings; teacher-only | Auxiliary/pretraining, no target leakage | P1 post-selection |
| MET/Q-GEM/atom-QM pretraining | Charges, Wiberg, geometry and electronic SSL | Reuses known geometry/torsion ideas but adds supervision | Separate pretrained teacher and transfer experiment | P2 |
| HOMO+LUMO+Gap multi-task | Related frontier-orbital targets | No new encoder channel | Post-selection target contract | P2 |
| GeoOpt-Net / conformer refinement | Better geometry input | Different from architecture; changes ETKDG source | New immutable train/inference geometry cache | Conditional on geometry residuals |
| nablaColors/FACET ensemble | Conformer fidelity + fragments | Different from one-ring token; multi-view teacher | Separate multi-conformer teacher | P2/P3 |
| QO2Mol/VQM24 | Larger or exhaustive quantum geometry corpora | Dataset/teacher only | OOD or pretraining benchmark | P3 |
| CELLI-style Qeq | Explicit global charge equilibration | More principled than arbitrary contacts | New long-range electronic contract | P3 |
| GW/BSE transfer | Multi-fidelity expensive targets | Separate from B3LYP Gap | `production/05_delta_gw`-type track | P2 when requested |
| EDBench / ED-DiT | Electron density, orbital energies, conserved electron number | Highest electronic relevance, but PCQM-derived overlap and basis mismatch | Train-role-only teacher after identity/level audit | P1 post-selection |
| MoiréGT / RadialFocus | Distance-modulated attention with an apparent 46 meV PCQM result | Explicit 3D coordinates / no official test-dev lineage | Negative control; never enter 2D candidate pool without a clean reproduction | Exclude |

## Recommended execution order

1. Finish and accept K3b under its existing protocol. Do not run a new paper-
   inspired architecture in parallel with the active causal comparison.
2. If K3b is below the material-gain gate, the only new random-init screen
   justified by this extension is a minimal MetaGIN-style path-count channel.
   Keep it static, sparse, graph-only, and below the 4M ceiling. Do not add
   MetaGIN's full width or call 3-hop path counts a learned 3D conformer.
3. If architecture selection is complete, prioritize a QCDGE/HEDMoL-style
   electronic teacher pilot. QCDGE is the first source to audit because its
   ground-state method is closest to the project target, but PubChemQC/PCQM
   overlap must be removed before any use. Compare readout-only transfer,
   frozen node features, and a no-teacher control under a new protocol.
4. Use OP, partial charges, and bond order as auxiliary targets before trying
   another topology. The target must be chemically meaningful, independently
   available, and measured for leakage and level-of-theory mismatch.
5. Open the geometry route only if residual analysis demonstrates that ETKDG
   geometry is the limiting factor. Any GeoOpt-Net-inspired source must be
   trained and inferred with that same source; it cannot be silently mixed with
   ETKDG.
6. Keep multi-task HOMO/LUMO/Gap, conformer ensembles, CELLI, and GW/BSE as
   separate post-selection tracks. They are useful project extensions, not
   reasons to weaken the current direct Gap screen.
7. Treat EDBench/ED-DiT as the strongest newly found electronic-teacher line,
   but do not use it until the PCQM identity/role leakage audit and basis-level
   accounting are complete.

## What this extension closes

- It does not justify another full-GPS width/depth sweep. The new frontier
  review and the direct PCQM evidence both reinforce paired cost accounting.
- It does not reopen persistent torsion, dense atom-bond dual streams, or
  arbitrary contact edges. Q-GEM and GeoOpt-Net use torsion/geometry as part
  of pretraining or refinement; that is a different causal question from the
  project's failed random-init torsion state.
- It does not promote functional-group, ring, fragment, or hypergraph numbers
  from QM9/MoleculeNet to PCQM evidence. Their value is as deterministic
  decomposition or teacher hypotheses.
- It does not alter the production registry or authorize official validation,
  test-dev, or full-data training.
