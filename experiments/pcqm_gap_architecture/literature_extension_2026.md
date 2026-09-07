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

### E16a. DFT features versus strict delta-learning in conjugated polymers

**Source.** [Liu, Yan, and Liu, Nanoscale 2025](https://pubs.rsc.org/en/content/articlehtml/2025/nr/d4nr03702b),
with [supplementary information](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66c23a02f3f4b05290464885/original/si-file.pdf)
and a public [MIT code/data repository](https://github.com/Liu-Group-UF/Machine-Learning-for-Accurate-Optical-Gap-Prediction-in-Conjugated-Polymers).

The study uses `1,096` experimental conjugated-polymer records and a `227`-
polymer no-retraining validation set. Its best XGBoost model combines an
oligomer B3LYP-D3/6-31G* HOMO--LUMO gap with ECFP6 and reports `MAE = 0.065
eV`, `R^2 = 0.77` for the experimental optical gap. The modified oligomer
construction removes alkyl side chains and extends the conjugated backbone;
the DFT-gap/optical-gap correlation rises from `R^2 = 0.15` for the unmodified
monomer to `0.51` after modification.

This is a completed, public *teacher-feature* protocol, but not strict
residual delta-learning: the DFT descriptor is concatenated with ECFP6 and the
model directly predicts the experimental target. For MolGap, this supplies a
clean control design: compare direct prediction, low-fidelity feature
concatenation, and an explicitly defined residual target under the same split,
geometry, and B3LYP/6-31G* target contract. Its optical labels and oligomer
construction must remain external.

### E16b. Frontier-orbital-family transfer with Chemprop

**Source.** [Meng et al., Journal of Chemical Physics 2026](https://pubmed.ncbi.nlm.nih.gov/42268043/),
DOI `10.1063/5.0333521`.

The accessible abstract and publisher preview describe Chemprop pretraining on
GFN2-xTB frontier-orbital properties of polymer trimers, followed by transfer
to chain/bulk bandgaps, ionization energy, and electron affinity. Reported test
MAEs are `0.246`, `0.269`, `0.169`, and `0.136 eV`, respectively, with
`R^2 > 0.90`; the authors also check chain-length scaling and inter-property
consistency and screen approximately `12 million` repeat units.

The transferable idea is to reuse separate HOMO-, LUMO-, and gap-pretrained
representations for related downstream frontier targets, then test physical
relations rather than only scalar error. However, the audited sources do not
expose a complete code/checkpoint/split/target-theory packet or the full text.
This is therefore a design reference, not evidence for a current PCQM
experiment.

### E16c. OPoly26: public polymer database with unresolved frontier-field contract

**Source.** [OPoly26](https://arxiv.org/pdf/2512.23117), the [official OMol25
model/data page](https://huggingface.co/facebook/OMol25), the [ColabFit train
record](https://materials.colabfit.org/id/DS_wfekwbgncjd3_0), the [OPoly26
validation schema](https://huggingface.co/datasets/colabfit/OPoly26-val), and
the [fairchem codebase](https://github.com/facebookresearch/fairchem).

OPoly26 is a large, public polymer-domain asset rather than a direct MolGap
method: the paper reports more than `6.35M` DFT calculations, more than `1.2B`
atoms, `94,000` amorphous-polymer cells, more than `239,000 ns` of MD, and
`2,444` unique monomers. The paper and official model page specify
`omegaB97M-V/def2-TZVPD`, while Appendix A lists frontier-orbital quantities
including HOMO energies and HOMO--LUMO gaps. The retrievable ColabFit train
record has `6,104,876` configurations and more than `1.1B` atoms, but its
description says `B97M-V/def2-SVP`, its method field says
`DFT-omegaB97M-V`, and its calculated-property summary advertises only
energy/forces. The validation schema exposes `electronic_band_gap` and
`def2-TZVPD` sample metadata. This is a real evidence conflict, not a detail
that can be filled in by assumption.

For MolGap, OPoly26 is therefore a public database/schema and polymer-OOD
reference. Its condensed-phase/MD/DFTB/AFIR geometries and
`omegaB97M-V/def2-TZVPD` labels do not satisfy the current ETKDG and
B3LYP/6-31G* contract. Before any future teacher or pretraining use, freeze
the exact release, reconcile field coverage and theory metadata, hash the
files, and quantify overlap with PCQM/Track A. No OPoly26 rows, frontier
fields, or weights enter the current database or experiment queue.

### E16d. PubChemQC-100K transfer to conjugated oligomers

**Source.** [Deng, Ng, and Li, Molecular Systems Design & Engineering
2025](https://pubs.rsc.org/en/content/articlehtml/2025/me/d4me00188e) and the
public [supporting information](https://www.rsc.org/suppdata/d4/me/d4me00188e/d4me00188e1.pdf).

This is a useful same-source transfer protocol. The authors select `106,429`
PubChemQC molecules with more than six double bonds and a gap below `6 eV`,
pretrain a width-128 SchNet, freeze its six interaction blocks, add one new
interaction block, and fine-tune on `610` conjugated oligomers (`131` monomer
types, polymerization degree `4--10`) at stated B3LYP/6-31G*. The target split
is `400/100/110`. Their direct-versus-transfer MAEs are `1.34/0.74 eV` for
HOMO, `0.67/0.46 eV` for LUMO, and `0.71/0.54 eV` for Gap; the same model
filters `3,710` candidates to `256` ML survivors and `46` DFT-validated
candidates.

The result supports target-distribution selection, frozen-backbone plus
one-block adaptation, and a direct-training control. It does not prove a
PCQM4Mv2 gain: the target is external CO-610, the coordinate construction is
not closed to MolGap's ETKDG contract, no official code/checkpoint was found,
and the filtered PubChemQC pool may overlap the repaired-2M lineage. Keep it as
a same-source teacher/pretraining protocol reference and require identity,
geometry, license, and role audits before any reuse.

### E16e. Public GFN2-xTB/COCONUT proxy workflow for gap and delta audits

**Source.** [Thinius et al., Digital Discovery 2026](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00186b),
the public [GitHub workflow and data repository](https://github.com/sthinius87/HL-gaps-pub),
and its persistent [Zenodo v0.2.1 archive](https://doi.org/10.5281/zenodo.15113790).

The paper builds a low-fidelity HOMO--LUMO-gap resource from approximately
`407,000` COCONUT natural products. It generates ten RDKit conformers per
molecule, optimizes them with GFN2-xTB/BFGS, computes same-level xTB orbital
gaps, and combines conformers with Boltzmann weights. A `70/30` split and
repeated shuffle-split evaluation produce a reported MLPR test MAE of
`0.210 +/- 0.001 eV` and RMSE of `0.298 +/- 0.002 eV`; XGBoost reaches about
`0.180 eV` on one split but shows a larger generalization gap. An external
evaluation on roughly `133,000` QM9 molecules demonstrates the theory shift:
the xTB-trained model underestimates the paper's B3LYP/6-31G(2df,p) DFT gaps by
about `3.94 eV` and GW gaps by `7.75 eV` on average.

This is unusually complete evidence for a proxy-generation workflow because
the paper points to versioned code, full calculated gaps/descriptors, CWL
definitions, and an archive rather than only a score. It is not, however, a
same-label delta-learning result for MolGap. Its target is GFN2-xTB, its
geometry path is RDKit plus xTB optimization, and its chemistry is COCONUT
rather than the official PCQM4Mv2 or repaired-2M PubChemQC role. The transferable
lesson is the audit shape: record proxy theory, conformer policy, aggregation,
cost, residual/error strata, and OOD theory shift before deciding whether a
proxy can reduce target error. A future same-PCQM delta route may borrow this
workflow shape only after recomputing the proxy under the exact ETKDG contract.

**Disposition.** Evidence level A for public low-fidelity workflow/data
provenance; B for a future CPU residual protocol; C for current MolGap labels,
weights, and database use. Do not report the xTB MAE as PCQM evidence.

### E16f. QMCVNet: PubChemQC PM6/MMFF geometry as a historical control

**Source.** [Maser and Reisman, CaltechAUTHORS record](https://authors.library.caltech.edu/records/2jygg-n1r30),
the attached [ChemRxiv paper PDF](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60ea947a9ab06e2e274d6cd7/original/3d-computer-vision-models-predict-dft-level-homo-lumo-gap-energies-from-force-field-optimized-geometries.pdf),
and [ChemRxiv DOI `10.33774/chemrxiv-2021-11r61`](https://doi.org/10.33774/chemrxiv-2021-11r61).

QMCVNet studies voxelized 3D CNNs for predicting B3LYP/6-31G* HOMO--LUMO
gaps from cheaper structures in the PubChemQC PM6 lineage. The paper uses a
`2.5M` B3LYP subset, about `1.8M` filtered molecules, and a `100k` architecture
screen followed by `1M` scaling. PM6 coordinates, RDKit MMFF coordinates, and
MMFF structures aligned to PM6 are all used, with an `80/10/10` split and
ten-fold right-angle rotation augmentation. On the `1M` screen, ShapeEncoder-d
reports `0.418 eV` MAE from PM6 coordinates after 20 epochs and `0.455 eV` from
MMFF after 100 epochs; the explicit PM6 electronic gap baseline is `0.400 eV`.

The study is informative but not a current method lead. It is a discussion
preprint with no audited code/checkpoint release, combines multiple geometry
contracts, and directly predicts the high-level gap rather than defining a
residual `Gap_B3LYP - Gap_PM6`. The source lineage may overlap Track A, and its
PM6/MMFF inputs violate the project's ETKDG train/inference rule. Its safe use
is a negative control template: compare a direct model with an explicitly
defined low-fidelity proxy, report coordinate and rotation costs, and keep the
proxy theory separate from the target.

**Disposition.** Evidence level B as historical same-lineage geometry/proxy
control; C for current database, initialization, labels, and code reuse.

### E16g. QUED: a complete electronic-descriptor teacher package

**Source.** [Hinostroza Caldas et al., Digital Discovery 2026](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00411j),
the [MIT-licensed implementation and model repository](https://github.com/lmedranos/QUED),
and the [Zenodo archive](https://doi.org/10.5281/zenodo.17106019).

QUED combines BOB or SLATM geometric descriptors with a named DFTB3+MBD
electronic descriptor containing global quantities, molecular-orbital energies,
and padded atom-level properties. It evaluates approximately `42k` equilibrium
and `42k` highly distorted QM7-X structures for PBE0+MBD properties, including
HOMO--LUMO Gap. The supplementary XGBoost table reports Gap MAE reductions from
`2.821` to `1.442` kcal/mol on the equilibrium subset and from `10.922` to
`4.071` kcal/mol on the distorted subset when SOAP is augmented with `D_QM`
(approximately `0.122 -> 0.063 eV` and `0.474 -> 0.177 eV`).

The public package includes descriptor generation, DFTB+ wrappers, CPU
training/evaluation scripts, model pickles, HDF5 training sets, and a notebook.
Its larger-molecule path is RDKit/MMFF followed by CREST/GFN2-xTB in GBSA water,
up to ten low-xTB-energy conformers, and DFTB3+MBD property calculation. This
is a high-quality teacher-feature engineering reference, but it is not a
same-target delta model: the target theory is PBE0+MBD/QM7-X or an ADMET
dataset, the electronic proxy is DFTB3+MBD, and the geometry does not satisfy
MolGap's ETKDG contract.

**Disposition.** Evidence level A/B for reproducible electronic-feature
packaging and geometry/electronic/combined ablations; C for current PCQM rows,
labels, weights, and direct feature use. If revisited, the safe borrowing is
the field-level teacher manifest and three-way ablation, recomputed under an
ETKDG-compatible, identity-audited proxy contract.

### E16h. POS-EGNN/OMol25: multi-task frontier-orbital pretraining with an explicit relation

**Source.** The peer-reviewed [EES Batteries paper](https://pubs.rsc.org/en/Content/ArticleLanding/2026/EB/D6EB00024J),
the [IBM/materials implementation](https://github.com/ibm/materials), and the
[Hugging Face POS-EGNN model card](https://huggingface.co/ibm-research/materials.pos-egnn).

The paper pretrains POS-EGNN on more than `20M` OMol25 electrolyte structures
at `ωB97M-V/def2-TZVPD`, predicting HOMO, LUMO, and Gap, then fine-tunes on
approximately `24k` lithium-electrolyte structures selected by composition,
hydration, and lithium-coordination rules. Its GotenNet-based equivariant
encoder uses a `6 Å` cutoff and max pooling. The four-target Huber loss gives
larger weights to HOMO and LUMO, a smaller weight to Gap, and adds a site-charge
head; the paper explicitly motivates this with `Gap = LUMO - HOMO`. The small
model reports held-out OMol25 MAEs of `0.489 eV` for HOMO and `0.606 eV` for
LUMO, with all three frontier targets below `0.61 eV` and correlations above
`0.9`.

The public package is useful but must be role-separated. IBM's Apache-2.0 repo
contains POS-EGNN code and an example notebook; the public `pos-egnn.v1-6M.pt`
weights are MPtrj energy/force/stress weights trained on `1.4M` samples, not the
paper's OMol25 frontier-orbital model. The paper's target theory, explicit MD
solvation/ion-pair geometries, and external OMol25 role are incompatible with
the current PCQM B3LYP/6-31G*/ETKDG contract.

**MolGap implication.** POS-EGNN is strong evidence for a future HOMO/LUMO/Gap
physical-consistency ablation and a public 3D foundation-model engineering
reference. It is not permission to import IBM weights, OMol25 rows, or explicit
solvation geometries into the current database.

**Disposition.** B for multi-task/physical-consistency and public 3D foundation
design; C for current labels, weights, rows, geometry, and pretraining.

### E16i. LUMIA: chemistry-informed organic-electronics pretraining and search

**Source.** The peer-reviewed [JCTC paper](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00713),
the [MIT-licensed implementation](https://github.com/YajingSun-Group/LUMIA),
and the [Zenodo data/weight release](https://zenodo.org/records/15852302).

LUMIA pretrains an RGCN on approximately `1.4M` organic molecules with
knowledge-informed edge and substituent masking intended to emphasize
π-conjugation and substituent effects. The repository exposes pretraining,
fine-tuning, fixed OCELOT folds, substructure-mask explanations, and an MCTS
workflow for searching attribution patterns; Zenodo exposes the pretraining
and downstream data archives plus a model dump with MD5 values. The published
scientific scope is organic optoelectronics, including OCELOT and reorganization
energy insight, not an official PCQM4Mv2 Gap benchmark.

**MolGap implication.** LUMIA is a completed engineering reference for
domain-informed 2D contrastive pretraining, explanation artifacts, and
knowledge-discovery audit trails. Its data, weights, knowledge masks, downstream
targets, and DGL/RGCN environment are external to the B3LYP/6-31G*/ETKDG
contract. The safe borrowing is the artifact/ablation pattern: freeze the
transformation definition, source rows, checkpoint hash, downstream folds, and
explanation outputs before asking whether a conjugation-aware teacher transfers.

**Disposition.** A/B for public organic-electronics pretraining and
interpretability engineering; C for current labels, weights, database, and
PCQM Gap evidence.

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

### E18a. Public 134k-molecule GW frontier-orbital teacher database

**Source.** [Accurate GW frontier orbital energies of 134 kilo
molecules](https://www.nature.com/articles/s41597-023-02486-4), with the
[open arXiv record](https://arxiv.org/abs/2303.08708) and the public
[Figshare archive](https://figshare.com/articles/dataset/Accurate_GW_frontier_orbital_energies_of_134_kilo_molecules_of_the_QM9_dataset_/21610077).

The dataset covers 133,885 QM9 molecules and provides PBE, G0W0, and
eigenvalue-self-consistent GW@PBE HOMO/LUMO quantities, with basis-set
extrapolated values. The records are keyed by the original QM9 identifiers and
the paper explicitly proposes the data for delta-learning and transfer
learning. The reported DFT-to-GW correlations are molecule-dependent, with
weaker LUMO correlation than HOMO, so a learned residual is more defensible
than a fixed energy shift.

**MolGap implication.** This is a strong public teacher-data reference, but it
is not a replacement for the current database: its targets are GW
quasiparticle energies, its source is QM9, and its theory/geometry contract is
different from B3LYP/6-31G* PCQM. Any use would require exact QM9-to-PCQM
identity/substructure auditing, train-role-only supervision, and a separate
theory declaration. Keep it as an external delta/transfer resource; do not
append it to Track A or Track B and do not count it as evidence for current
PCQM Gap improvement.

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

## I. Direct same-database self-supervised pretraining

### E27. Pre-training via Denoising (PVD)

**Primary sources.** The ICLR 2023 paper is
[Pre-training via Denoising for Molecular Property
Prediction](https://arxiv.org/html/2206.00133), with an
[MIT-licensed official repository](https://github.com/shehzaidi/pre-training-via-denoising)
and a documented [PCQM4Mv2 configuration](https://github.com/shehzaidi/pre-training-via-denoising/blob/main/examples/ET-PCQM4MV2.yaml).

PVD perturbs equilibrium 3D coordinates with isotropic Gaussian noise and
trains an equivariant vector head to predict the injected noise. Its score-
matching derivation interprets the result as an approximate force field around
the observed equilibrium structures. The paper explicitly mean-centers the
noise because a global translation is unidentifiable. This is a self-supervised
structural objective, not a force label, a low-fidelity electronic target, or
strict delta-learning.

The upstream is `3,378,606` PCQM4Mv2 DFT-equilibrium structures and the labels
are not used. The paper's QM9 table reports three-seed GNS-TAT improvements
from `17.3/17.1/25.7` meV to `14.9/14.7/22.0` meV for HOMO/LUMO/Gap after
PCQM denoising pretraining on top of the same Noisy-Nodes family. Its isolated
TorchMD-NET control separates random initialization (`22.0/18.7` meV), a
downstream denoising auxiliary task (`18.1/15.6`), and upstream PCQM
pretraining (`15.6/13.2`) for HOMO/LUMO. The official repository includes a
retrievable `denoised-pcqm4mv2.ckpt`, the PCQM pretraining command, and a QM9
fine-tuning command/config.

The paper also shows why this is not a free scaling recipe: upstream-size
benefit saturates, downstream gains increase when labels are scarce, and PCQM
pretraining does not improve final OC20 IS2RE validation when the upstream and
downstream distributions differ substantially. A frozen-backbone probe still
beats a random frozen backbone, but full fine-tuning remains better.

**MolGap disposition.** This closes an important evidence gap: there is a
completed, code-backed precedent for direct PCQM structure pretraining and a
causal control separating pretraining from an auxiliary denoising loss. It is
not a current experiment because its coordinates are DFT equilibrium rather
than ETKDG, its upstream role is the full PCQM structure pool, and it reports
QM9/OC20 downstream outcomes rather than a same-contract PCQM Gap result. A
future ETKDG-only audit could borrow mean-centered vector denoising, the three-
way control, and upstream compatibility curves, but cannot import its
checkpoint, coordinates, or metrics.

See the [full PVD deep-reading card](deep_reading_denoising_continuation_2026-09-07.md#5-pre-training-via-denoising-pvd-the-clean-pcqm-pretraining-baseline)
for the separate TorchMD/GNS configurations and geometry audit.

### E28. Fractional Denoising (Frad/FradNMI)

**Primary sources.** The ICML 2023 formulation is
[Fractional Denoising for 3D Molecular Pre-training](https://arxiv.org/html/2307.10683)
with [official code](https://github.com/fengshikun/Frad). The expanded Nature
Machine Intelligence paper is
[Pre-training with Fractional Denoising to Enhance Molecular Property
Prediction](https://arxiv.org/html/2407.11086), with
[FradNMI code](https://github.com/fengshikun/FradNMI), [Zenodo weights](https://zenodo.org/records/12697467),
and [Figshare source data](https://doi.org/10.6084/m9.figshare.25902679.v1).

Frad introduces a chemical-aware noise (CAN) stage before coordinate Gaussian
noise (CGN): `x_eq -> x_med -> x_fin`. It only predicts `x_fin - x_med`, the
CGN component. This allows CAN to model rotatable-bond torsions (RN) or
bond-length/bond-angle/torsion vibrations (VRN) without losing the theorem's
force-learning interpretation, which relies on the final conditional noise
being isotropic Gaussian. The NMI PCQM recipe uses batch `70`, AdamW, `10,000`
warmup steps, maximum learning rate `4e-4`, a `400,000`-step cosine cycle,
RN torsion scale `2`, VRN scales `0.058/0.129/0.18/1`, and CGN standard
deviation `0.04`.

The upstream is the same `3,378,606`-structure PCQM4Mv2 pool, with labels
unused. On QM9, coordinate denoising gives HOMO/LUMO/Gap
`17.7/14.3/31.8` meV; Frad(RN) gives `15.3/13.7/27.8` and Frad(VRN)
`17.9/13.8/27.7`. The paper reports 9/12 QM9 targets at a new best and
improvement over the same TorchMD-NET backbone on 11/12 targets. Its force,
MD17/MD22/ISO17, LBA, and inaccurate-conformer studies support the mechanism,
but none is a same-contract PCQM Gap result. The inaccurate-conformer test is
specifically RDKit Distance Geometry plus MMFF; it does not establish
equivalence to MolGap ETKDG.

**MolGap disposition.** Frad is a complete chemical-aware pretraining
reference with code, weights, source data, theory, and matched ablations. It
is not a current candidate: its main PCQM coordinates are DFT equilibrium,
its robustness control is RDKit+MMFF, its frontier results are QM9, and the
project's random-init torsion-state route is already closed. A future
post-selection ETKDG-only audit may borrow the intermediate/final-coordinate
manifest and the coordinate-denoising/no-pretraining controls, but cannot
import Frad weights, coordinates, or metrics.

See the [full Frad deep-reading card](deep_reading_denoising_continuation_2026-09-07.md#6-fractional-denoising-frad-chemical-aware-noise-without-breaking-the-score-objective)
for the theorem, ring/degree handling, robustness, and cost audit.

### E29. Sliced Denoising (SliDe)

**Primary sources.** [ICLR 2024 paper](https://arxiv.org/html/2311.02124),
[official MIT code](https://github.com/fengshikun/SliDe), and the
[ICLR proceedings record](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a1d69d1f64c6b6df105b15984ca527a-Abstract-Conference.html).

SliDe derives a quadratic BAT energy over bond lengths, bond angles, and
torsions from Open Force Field 2.0.0 parameters. It samples chemically
structured relative-coordinate noise and uses Gaussian random slicing plus
finite coordinate differences to avoid explicit Cartesian/BAT Jacobians. The
GET encoder adds angle/torsion-aware edge updates to a TorchMD-NET-like
equivariant Transformer. On 1,000 PCQM4Mv2 molecules, its estimated-force
Pearson correlation is `0.895(0.071)`, versus `0.616(0.047)` for coordinate
denoising and `0.631(0.046)` for Frad. The matched QM9 table reports
HOMO/LUMO/Gap `13.6/12.3/26.2` meV, versus Coord `17.7/14.3/31.8` and Frad
`15.3/13.7/27.8`; the pretraining ablation reports scratch
`17.6/16.7/31.3`, without regularization `15.0/14.8/27.7`, and with
regularization `13.6/12.3/26.2`.

The PCQM pretraining pool is about `3.4M` label-free equilibrium structures.
The public recipe exposes batch `128`, AdamW, `10,000` warmup steps, maximum
learning rate `4e-4`, cosine cycle `240,000`, `N_v=128`, `sigma=0.001`, and
coordinate regularization scale `tau=0.04`; the authors report eight A100
GPUs. These results are not a PCQM4Mv2 Gap score: the main coordinates are
DFT equilibrium structures and the frontier-orbital downstream table is QM9.

**MolGap disposition.** Strong B-level physics-informed pretraining and code
reference; no current experiment, checkpoint initialization, database change,
or reopening of the closed torsion-state route. A future adaptation must use
ETKDG at both stages, freeze BAT parameter/ring/degeneracy rules, and compare
random-init, coordinate-denoising, and SliDe-style objectives at the same
student capacity. See the [full SliDe deep-reading card](deep_reading_denoising_continuation_2026-09-07.md#7-slide-force-consistent-bondangletorsion-pretraining-with-random-slicing).

### E30. Coordinating Cross-modal Distillation (CCMD)

**Primary source.** [Coordinating Cross-modal Distillation for Molecular
Property Prediction](https://arxiv.org/html/2211.16712).

CCMD trains a DFT-coordinate 3D Graphormer teacher and distills it into a 2D
Graphormer student. It separates all-layer virtual-token (global molecular)
alignment from atom-token (local) alignment and derives a molecular-size
normalization, with order `1/N^2` for a Transformer. The PCQM validation table
reports Graphormer `0.0864`, APE baseline `0.0845`, global distillation
`0.0822`, global plus manually searched local weighting `0.0818`, and global
plus size-coordinated local weighting `0.0809` eV. Naive local alignment alone
degrades to `0.0870`; all-layer and size-normalized controls are therefore
more informative than the headline. The abstract separately states `0.0734`
on the 2022 test-challenge, but the accessible main table is a validation
comparison and no verified code/checkpoint was found.

**MolGap disposition.** A for global/local teacher-loss and negative-transfer
evidence, B/C for current use. DFT teacher coordinates, a prior Graphormer
split protocol, 68M-scale backbone, and absent executable artifact prevent
direct admission. Preserve it as the control design for a separately
authorized ETKDG teacher study; do not treat the `0.0809` or `0.0734` values as
current MolGap results. See the [full CCMD deep-reading card](deep_reading_teacher_delta_continuation_2026-09-07.md#2a-coordinating-cross-modal-distillation-ccmd-globallocal-teacher-loss-with-size-normalization).

## Decision matrix for the remaining search

| Route | New information | Overlap with closed routes | Contract class | Priority |
|---|---|---|---|---:|
| MetaGIN-style path counts | Static 2D 2/3-hop path/conjugation statistics | Low; not a coordinate torsion state | One seed-42 random-init screen, `<=4M`, same 100K/10K split | P1 after K3b |
| QCDGE auxiliary teacher | Near-target B3LYP/6-31G* HOMO/LUMO, Mulliken, excited-state labels | Low, but source overlap risk is high | Frozen external teacher; identity dedup and level-of-theory audit | P1 post-selection |
| HEDMoL/OP/charge teacher | Substructure, bond, or atom electronic descriptors | Distinct from contacts/rings; teacher-only | Auxiliary/pretraining, no target leakage | P1 post-selection |
| MET/Q-GEM/atom-QM pretraining | Charges, Wiberg, geometry and electronic SSL | Reuses known geometry/torsion ideas but adds supervision | Separate pretrained teacher and transfer experiment | P2 |
| EMPP | Label-free masked-position 3D pretraining on PCQM | Requires a 3D coordinate contract and pretrained initialization | Post-selection objective reference; no direct PCQM Gap gain | P2 |
| Suiren-1.0 | Large 3D teacher plus frozen 2D conformation distillation | External B3LYP/def2-SVP data, 1.8B scale, non-ETKDG geometry | Staged teacher/student design reference only | P2 |
| Uni-3DAR | Octree/subtree-compressed autoregressive 3D modeling with a public QM9/DRUG/MP20 implementation; SpaceFormer 20K HOMO/LUMO/Gap setting | External 19M 3D corpus, SpaceFormer task/split, non-ETKDG geometry, and no direct official PCQM4Mv2 Gap result | Hierarchical geometry-tokenization reference only | P2 |
| QCML | 14.7B low-fidelity plus 33.5M PBE0 multi-fidelity data, outlier/status fields | Off-equilibrium UFF/xTB geometry and PBE0 energy/force/matrix targets | Schema and transfer-learning design reference; no bulk merge | P2/P3 |
| qcMol | 1.2M molecules with global Gap and local atom/bond descriptors | B3LYP-D3/def2-SV(P)//GFN2-xTB, single geometry, mixed-source overlap | Identity-filtered electronic teacher candidate; no current initialization | P1 post-selection |
| QM40 | 162,954 B3LYP frontier-orbital records for 10--40-atom drug-like molecules | 6-31G(2df,p), xTB/DFT geometry, ZINC distribution | Near-target external size/scaffold audit; no concatenation | P2 |
| QeMFi | Five-fidelity target/cost benchmark with public scripts | Nine molecules, TD-DFT excitations, Wigner/geodesic geometry | Delta-learning protocol reference only | P3 |
| PubChemQC B3LYP/6-31G*//PM6 | Same-origin 86M electronic release with orbital energies and queryable HOMO/LUMO/Gap fields | PM6-optimized geometry, broad 2016 PubChem snapshot, exact overlap/release audit absent | Closest same-origin teacher/proxy source; no merge or initialization | P2/P3 |
| MFΔML | Public Δ-ML/MFML/MFΔML comparison with explicit training and inference cost | QeMFi basis-set fidelities, KRR/Coulomb-matrix descriptor, nine molecules | Residual/cost gate for any future same-database delta route | P2 |
| ViSNetGWBSE | OMol25/QCDGE low-fidelity pretraining, full vs readout-only transfer, qsGW/GW-BSE learning curves, public code/checkpoints | External functionals/bases, qsGW/BSE targets, explicit Cartesian geometry, not current B3LYP/6-31G* Gap | Multi-fidelity transfer and outlier/data-demand protocol reference; no current initialization | P2 |
| MoleculeSDE / GraphMVPv2 | PCQM paired 2D/3D pretraining, data-space 2D↔3D SDEs, public code and checkpoints | PCQM paired conformers, old environment, QM9 downstream rather than direct PCQM Gap | ETKDG-only denoising/pretraining design reference; no current initialization | P2 |
| MoleculeJAE | Joint 2D/3D trajectory score matching plus contrastive surrogate; QM9 Gap/HOMO/LUMO ablations | No verified code/checkpoint, PCQM paired geometry, QM9 rather than target-matched PCQM score | ETKDG trajectory-objective reference after architecture selection | P2/P3 |
| MoleBlend | Relation-level SPD/edge/3D blending, modality-targeted relation prediction, public code | Legacy stack, PCQM Gap mentioned without a directly checkable score/split table, non-ETKDG pretraining geometry | Pair-relation pretraining design reference only | P2 |
| FlexMol | Paired-PCQM pretraining plus missing-modality decoders and single-modality continuation | Stage 2 imports Uni-Mol data; PCQM DFT geometry; no direct PCQM Gap result | ETKDG student/teacher and missing-modality contract reference | P2 |
| DenoiseVAE | Molecule-adaptive atom-wise noise distributions, public ICLR 2025 code, reported PCQM4Mv2 Gap validation `0.0777 +/- 0.0005` with `1.44M` parameters | Exact split/coordinate path/checkpoint/license and ETKDG compatibility are not yet independently closed | Highest-priority evidence-only pretraining audit; no warm start or run yet | P1 audit |
| PVD | Direct PCQM4Mv2 self-supervised denoising, official checkpoint/config, QM9 HOMO/LUMO/Gap transfer, and random-init/Noisy-Nodes/pretraining controls | DFT-equilibrium coordinates, full-PCQM upstream role, TorchMD/GNS stack, and no same-contract PCQM Gap result | Completed denoising reference; future ETKDG-only protocol after architecture selection | P1 audit/reference |
| Frad / FradNMI | Chemical-aware RN/VRN noise, fractional CGN target, public code/weights/data, QM9 frontier and force/robustness ablations | DFT main coordinates, RDKit+MMFF robustness geometry, legacy TorchMD stack, QM9 downstream, and closed torsion-state route | Chemical-aware pretraining reference; future ETKDG-only audit only | P1 audit/reference |
| SliDe | BAT bond/angle/torsion pretraining, random-sliced force objective, public code/models, PCQM force audit and QM9 frontier ablations | DFT equilibrium geometry, OpenFF/Sage prior, QM9 rather than PCQM Gap downstream, GET/Nv cost | Future ETKDG-only physics-informed pretraining reference | P1 audit/reference |
| CCMD | Direct PCQM validation of global/local 3D-to-2D distillation and size-normalized atom loss | DFT teacher coordinates, prior split, large Graphormer, no verified code/checkpoint | Future ETKDG teacher-loss control design; no current score | P1 teacher reference |
| 3D-GSRD | Selective re-mask decoding, public PCQM pretraining and QM9 `homo/lumo/gap` fine-tune scripts | No directly checkable official PCQM Gap score; 3D/DFT geometry and budget differ | Decoder leakage-control reference | P2 |
| 3D-MolT5 | Discrete local 3D tokens, PCQM 3D pretraining, PubChemQC Gap ablation `0.0791` vs `0.0968` without 3D | PubChemQC/text setting, external corpora, and non-ETKDG geometry are not the official PCQM contract | Compact masked-geometry auxiliary-task reference | P2 |
| MolSpectra | PCQM denoising plus QM9Spectra UV--Vis/IR/Raman teacher; QM9 Gap `26.8` vs `31.8` meV coordinate baseline | B3LYP/def-TZVP spectral labels, QM9 identity, geometry, and repository license need audit | Electronic teacher/auxiliary-objective reference | P1 post-selection |
| 3D-PGT | Direct paper PCQM4Mv2 validation `0.0762` with 42.6M GPS-style parameters; automated bond/angle/dihedral pretext fusion | DFT equilibrium pretraining geometry, large model, paper validation protocol | Historical direct-PCQM pretraining reference; possible reduced ETKDG contract only after a new protocol | P1 audit |
| AniDS | Public PCQM label-free pretraining, atom-wise full-covariance anisotropic noise, 8.9% MD17 force improvement | 129M Equiformer-scale force model, DFT coordinates, no direct Gap result | Adaptive-noise design reference after scalar-noise audit | P2 |
| 3D-EMGP | Public equivariant force/noise-scale pretraining with QM9 `gap/homo/lumo` fine-tuning | GEOM-QM9 rather than PCQM, old stack, no direct PCQM Gap result | Historical physical-objective reference | P2 |
| Mol-MFFGE | Task-aware learnable noise transformation and bi-level weighting of denoising plus downstream losses; public code with QM9 `homo/lumo/delta` paths | GEOM/SPICE/QM9/MD17 rather than PCQM/ETKDG; no direct PCQM Gap result and no independently verified checkpoint | Task-aware denoising reference after a future ETKDG-only contract | P2 |
| OCNet | Conjugated-domain SE(3) pretraining on 10M-scale molecular/dimer assets, TB electronic descriptors, public weights, and OCELOT H-L Gap MAE `0.008 eV` | OCELOT/OCNet theory, geometry, dimer/film roles, generated chemistry, and non-ETKDG inputs | Organic-electronics teacher/OOD and conjugation-error stratification; no label merge | P1 post-selection |
| LUMIA | ~1.4M-molecule chemistry-informed contrastive RGCN pretraining, public data/weights, OCELOT/optoelectronic fine-tuning, substructure explanation, and MCTS search | External organic-optoelectronic roles, no directly verified PCQM Gap score, DGL/RGCN environment, and no ETKDG/B3LYP contract | Organic-domain pretraining and interpretability/artifact reference; no current labels or weights | P2 |
| DFT-to-experiment frontier-orbital transfer | XGBoost/Klekota--Roth transfer from 11,626 DFT rows to 1,198 experimental rows; HOMO/LUMO correlations `0.75/0.84` | Experimental targets and fingerprint/tree model are not PCQM B3LYP/6-31G* | Theory-to-experiment calibration protocol only | P2 |
| Conjugated-polymer D-MPNN pretraining | Three-way comparison of direct training, monomer-DFT pretraining, and TD-DFT-extrapolated polymer pretraining; own-task final gap MAE `0.074 eV` | Domain-matched proxy selection and full-fine-tuning ablation | Experimental polymer targets, TD-DFT/MMFF94s geometries, external data, and cited artifact URL currently unresolved | P2 |
| DFT-feature-assisted optical-gap transfer | Public paper/SI/repo; modified-oligomer DFT gap + ECFP6 gives own-task optical-gap MAE `0.065 eV` | Separate teacher feature from strict residual delta; nested-CV and group extrapolation | Experimental optical targets and external oligomer/DFT geometry; not PCQM Kohn--Sham Gap | P2 |
| Frontier-orbital Chemprop transfer | Published 2026 abstract/preview reports GFN2-xTB-trimer transfer to chain/bulk gap, IE, and EA | Frontier-family transfer and physical-consistency checks | No verified code, checkpoint, split, data identity, or complete target-theory packet | P3 |
| OPoly26 | Public multi-million-polymer DFT/MD database, fairchem path, ColabFit record, and validation schema with frontier-field evidence | Polymer condensed-phase/MD geometry, omegaB97M-V/def2-TZVPD theory, unresolved train metadata/field coverage, and no current PCQM Gap result | External database/schema/OOD reference; no data merge or current teacher | P2 audit |
| PubChemQC-100K -> CO-610 SchNet transfer | Published same-source B3LYP/6-31G* transfer, public ESI, filtered pretraining rule, frozen layers, added interaction block, and direct control | External oligomer target, no official code/checkpoint, unclear ETKDG geometry, and possible Track A/PubChemQC identity overlap | Same-source transfer protocol reference; no labels, weights, or filtered rows merged | P2 audit |
| GFN2-xTB/COCONUT proxy workflow | Complete public low-fidelity xTB gap/descriptors workflow, archived data, conformer aggregation, and external theory-shift test | GFN2-xTB target, RDKit/xTB geometry, natural-product distribution, and no B3LYP/6-31G*/ETKDG residual evidence | CPU proxy-generation and delta-cost reference only; exact ETKDG adaptation and residual/cost audit required | P2 audit |
| QMCVNet / PubChemQC PM6-to-B3LYP control | Same-lineage low-fidelity geometry comparison, voxel CNNs, explicit rotation augmentation, and direct PM6/MMFF controls | Discussion preprint, no audited code/checkpoint, mixed PM6/MMFF geometry, possible Track A overlap, and no strict residual target | Historical proxy/rotation negative control only; recompute under ETKDG before any future use | P3 |
| QUED electronic descriptor | Public DFTB3+MBD electronic descriptor, BOB/SLATM geometry, QM7-X Gap ablations, MIT code/models, and Zenodo archive | QM7-X/PBE0+MBD and ADMET roles, RDKit/MMFF+CREST/GFN2-xTB geometry, and no PCQM/B3LYP/6-31G*/ETKDG result | Electronic-teacher packaging and three-way ablation reference; no current feature/row/weight import | P1 audit |
| POS-EGNN/OMol25 frontier workflow | >20M OMol25 pretraining structures; HOMO/LUMO/Gap/site-charge multi-task loss; Huber weighting treats Gap as a physical consistency term; public IBM code and MPtrj weights | OMol25 `ωB97M-V/def2-TZVPD`, explicit MD solvation geometry, public weights target MPtrj energy/force/stress, and no PCQM/ETKDG result | Multi-task physical-consistency and 3D-foundation reference; no external rows or weights | P1 post-selection reference |
| AEGCNN-MTL | QM9 correlation-grouped HOMO/LUMO/Gap multi-task result plus within-architecture negative-transfer comparison | QM9/BDG roles, no public code/BDG data, and no direct PCQM/ETKDG result | Auxiliary-target/task-grouping control only | P2 reference |
| OSCs_RGGN | Public repository claims 48,182 OSC samples and RGNN predictions | No auditable dataset/license/split/theory manifest or released checkpoint | Quarantine discovery lead; no evidence or data import | C |
| GLACIER | Public KDD 2026 multimodal student--teacher model; 100K Enamine pretraining, graph/SMILES/descriptor fusion, MiniMol/MolFormer distillation, public code and checkpoint | TDC/MoleculeNet only; no PCQM Gap result; external corpus and descriptor/teacher overlap require a new role audit | Multi-teacher embedding cache, contribution floor, and frozen-teacher ablations after architecture selection | B engineering |
| QM9 embedding KD scalability study | Open-access 2025 paper and code; QM9 teacher includes HOMO/LUMO/Gap, smaller students, latent L1+cosine KD, and cross-domain transfer | QM9/ESOL/FreeSolv, relative `R^2`, Python 3.9, and no PCQM/ETKDG result | Teacher/no-teacher, student-capacity, and embedding-alignment control design only | B teacher reference |
| EDG electron-density teacher | IJCAI paper, MIT code/checkpoints, EDBench/PCQM-derived 2M density pretraining, and QM9 HOMO/LUMO/Gap ablations | Density basis, source conformers, PCQM row overlap, OneDrive artifacts, and non-ETKDG geometry are not closed for MolGap | Three-stage electronic teacher, frozen feature artifact, and geometry-student distillation reference | B teacher reference |
| ChemBERTa-3 | Public open training/benchmark framework, model variants, environment, data preparation, and Zenodo/repository artifacts | ZINC20/PubChem pretraining and no directly verified PCQM Gap evidence; external source and 3D conformer roles differ | Reproducible configs, benchmark scripts, model-size/featurizer ledger | B engineering |
| ChemFM | 3B-scale causal SMILES foundation model trained on 178M UniChem SMILES with public code/weights | No matched PCQM Gap result, no ETKDG path, and model/data scale exceeds the bounded screen | Scaling-report and corpus/version accounting only | B/C reference |
| GPSE | Public learned positional/structural encoder, six PSE families, and a retrievable PCQM-named checkpoint | PSE reconstruction is not a Gap result; checkpoint lineage, labels, split, and old PyG stack need audit | Frozen structural teacher or representation probe after provenance gate | B teacher |
| CondPSE | 2026 polynomial-filtered structural encoder; strong CSL/EXP synthetic gains over GPSE | Comparable to GPSE with no consistent molecular-property advantage; no official code | Negative control against admitting structural pretraining from synthetic expressivity alone | B negative control |
| GCPE | 2026 publisher page describes spatial/spectral/subgraph PE and PCQM4Mv2 experiments | Exact PCQM metric, split, code, checkpoint, and complete accessible paper were not verified | Observation only; no candidate or headline score | C |
| Chemprop benchmark v2 | MIT benchmark package with `qm9_gap` and `pcqm4mv2`, Zenodo data, explicit `data.csv`/`splits.json`, and Chemprop v2.0.3 environment | It is a baseline/engineering package, not evidence of a new MolGap architecture or a replacement split | Independent data-loader/metric/artifact sanity reference | B engineering |
| ECMMR | 2026 abstract describes BRICS hypergraphs, contrastive/generative cross-modal tasks, PCQM pretraining, and 22 downstream tasks | Exact Gap table, split, code, and checkpoint were not exposed in the audited source | Index hit only; no possible experiment | C |
| HOMO+LUMO+Gap multi-task | Related frontier-orbital targets | No new encoder channel | Post-selection target contract | P2 |
| GeoOpt-Net / conformer refinement | Better geometry input | Different from architecture; changes ETKDG source | New immutable train/inference geometry cache | Conditional on geometry residuals |
| nablaColors/FACET ensemble | Conformer fidelity + fragments | Different from one-ring token; multi-view teacher | Separate multi-conformer teacher | P2/P3 |
| QO2Mol/VQM24 | Larger or exhaustive quantum geometry corpora | Dataset/teacher only | OOD or pretraining benchmark | P3 |
| CELLI-style Qeq | Explicit global charge equilibration | More principled than arbitrary contacts | New long-range electronic contract | P3 |
| GW/BSE and multi-fidelity transfer | ESA and ViSNetGWBSE show DFT/TDDFT-to-GW/BSE transfer; MFGP-GEM, multi-fidelity GNNs, and dataset embeddings add autoregression, proxy/embedding/readout controls, and auditable HOMO/LUMO evidence | Separate from B3LYP Gap; QM9GWBSE/GW, QMugs, MultiXC/MatPES, or HF/MP2/CCSD roles and explicit 3D geometry differ | `production/05_delta_gw`-type track; split/identity/theory audit first | P2 when requested |
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
7. Treat EDBench/EDG/ED-DiT as the strongest newly found electronic-teacher
   line, but do not use any of them until the PCQM identity/role leakage audit,
   basis-level accounting, and exact ETKDG input contract are complete. EDG's
   released teacher/feature artifacts make it the first package to inspect when
   that audit is authorized; this is not an authorization to import them.

## Deep-reading continuation recorded on 2026-09-07

The subsequent primary-source reading is split into three auditable batches:

- [direct PCQM and attention](deep_reading_direct_pcqm_attention_2026-09-07.md):
  controlled global-attention evidence, GAPE, GRPE, TetraGT, Edge-Set
  Attention, GEM-2, TokenGT, GPTrans, Edge Transformer, quantum-computed
  positional encodings, GFSA, Specformer, and the (2,1)-GT/WL-Transformer
  reference, with explicit geometry and split audits; the AdvSynGNN index hit
  is recorded as a non-comparable negative control because its PCQM protocol is
  incomplete;
- [hierarchy, pretraining, and electronic teachers](deep_reading_hierarchy_pretraining_2026-09-07.md):
  RingFormer, SCAGE, EPT, M2UMol, TGF-M, SIMG, GraphGPT, GTAM, MolGroup, and
  TMP; TMP is a high-signal 3D pretraining reference but is not ETKDG-compatible
  as published;
- [foundation, geometry, and higher-order graphs](deep_reading_foundation_geometry_higher_order_2026-09-07.md):
  MIST, UMA, MotiL, GeoMFormer, GotenNet, EquiHGNN, GoMS, 3DMSE, the QMOF
  Molecular Graph Transformer, SpaceFormer, MolCL-SP, EMPP, and Suiren-1.0;
  EMPP is a masked-position pretraining reference, Suiren is a large
  teacher/CCD distillation reference, and Uni-3DAR is an octree/subtree
  geometry-tokenization reference; all use non-ETKDG or non-official-PCQM
  property contracts.
- [teacher/distillation and delta-learning continuation](deep_reading_teacher_delta_continuation_2026-09-07.md):
  a public PCQM 3D-prior distillation implementation, the D&D training and
  geometry contract, DelFTa, selected HOMO-LUMO delta-QML, and Δ-DFT/GW
  correction evidence. It records a synthetic-smoke failure as a constraint,
  not as a positive PCQM result.
- [electronic and delta-learning deep reading](deep_reading_electronic_delta_2026-09-07.md):
  atom-level quantum pretraining, HEDMoL/Q-GEM/overlap-population teachers,
  DelFTa-style residuals, the public 134k-QM9 GW frontier-orbital database,
  the ESA DFT-to-GW transfer control, the MFGP-GEM multi-step
  HOMO/LUMO benchmark, VQM24, QCML, qcMol, and QM40. These public resources remain
  teacher/OOD/protocol references only; none is added to the current PCQM or
  repaired-2M database roles.
- [multi-fidelity delta continuation](deep_reading_teacher_delta_continuation_2026-09-07.md):
  QeMFi's five-level TD-DFT benchmark and its explicit compute-time accounting,
  MFΔML's public residual/ensemble cost comparison, plus ViSNetGWBSE's
  OMol25/QCDGE-to-qsGW transfer, full-vs-readout comparison, and public
  QM9GWBSE release, are recorded as delta/transfer protocol references, not
  current Gap results. The same-origin PubChemQC 86M B3LYP/6-31G*//PM6
  release is separately recorded as a lineage lead, not an added database.
- [2D/3D trajectory and relation-pretraining continuation](deep_reading_2026-09-07.md):
  MoleculeSDE/GraphMVPv2, MoleculeJAE, MoleBlend, and FlexMol are now read at
  the paper-and-code level. Their public PCQM use is verified, but their
  strongest numeric results are QM9, MoleculeNet, or conformation-generation
  results rather than a directly checkable PCQM Gap score under this project's
  ETKDG/role contract.
- [adaptive denoising, discrete geometry, and electronic spectra](deep_reading_pretraining_electronic_2026-09-07.md):
  DenoiseVAE, 3D-GSRD, 3D-MolT5, and MolSpectra are read at the
  paper-and-code level. DenoiseVAE reports the only new direct PCQM Gap
  validation number in this batch, but its ETKDG/split/checkpoint contract is
  not yet closed; 3D-GSRD and 3D-MolT5 remain design references, and MolSpectra
  remains an external-theory electronic-teacher reference.
- [denoising continuation](deep_reading_denoising_continuation_2026-09-07.md):
  3D-PGT, AniDS, and 3D-EMGP are read at the paper-and-code level. 3D-PGT's
  `0.0762` is a direct paper PCQM validation number but uses DFT 3D pretraining
  and a much larger GPS model; AniDS and 3D-EMGP are adaptive-noise and
  physics-objective references without direct PCQM Gap results. Mol-MFFGE adds
  task-aware learnable noise transformation and bi-level auxiliary-loss
  weighting, but its evidence is GEOM/SPICE/QM9/MD17 rather than PCQM/ETKDG.
- [organic-electronics domain models](deep_reading_organic_electronics_2026-09-07.md):
  OCNet, LUMIA, a DFT-to-experiment frontier-orbital transfer model, and
  OSCs_RGGN are read separately. OCNet and LUMIA are reusable external
  pretraining/interpretability references for conjugated chemistry, but their
  OCELOT/TB/DFT/dimer or RGCN/optoelectronic roles are not PCQM labels;
  OSCs_RGGN remains a weak provenance lead.
- [student--teacher, foundation, and structural pretraining](deep_reading_foundation_teacher_structural_2026-09-07.md):
  GLACIER, the 2025 QM9 embedding-KD scalability study, EDG/EDBench,
  ChemBERTa-3, ChemFM, GPSE, CondPSE, GCPE, and the Chemprop v2 benchmark
  package are read at the paper/code level. EDG is the clearest new
  electronic-density teacher package; GLACIER is the clearest new multi-teacher
  engineering template; the QM9 KD study supplies a teacher/no-teacher control
  matrix; GPSE/CondPSE provide matched positive and negative evidence for
  learned structural encodings. None clears the current ETKDG/PCQM Gap
  admission gate, and ECMMR remains observation-only.
- [proxy and delta continuation](deep_reading_electronic_delta_2026-09-07.md):
  the GFN2-xTB/COCONUT workflow is recorded as a complete public low-fidelity
  asset and cost/audit reference. Its xTB target, RDKit/xTB geometry, and
  natural-product role remain external; it does not add rows, weights, or a
  current teacher to MolGap.
- The same continuation records QMCVNet as a historical PubChemQC
  PM6/MMFF-to-B3LYP geometry control and QUED as a public DFTB3+MBD electronic
  descriptor package. QMCVNet remains a mixed-contract negative control; QUED
  remains a teacher-feature/ablation reference because neither provides a
  legal B3LYP/6-31G*/ETKDG feature path for the current database.
- It also records POS-EGNN/OMol25 as a complete multi-task frontier-orbital
  design reference with public IBM code and MPtrj weights, while separating
  those weights from the paper's unreleased OMol25 frontier checkpoint. Its
  theory and explicit solvation geometry remain external. AEGCNN-MTL supplies
  independent QM9 evidence that task relatedness should be measured and that
  weakly coupled auxiliary targets can produce negative transfer; neither
  source authorizes a current experiment.

These batches add evidence and exclusions only. They do not change the
database, conformer method, active experiment, seed budget, or authorization
boundary. In particular, MIST is retained as a future SMILES-pretraining
teacher; EPT, UMA, GeoMFormer, GotenNet, EquiHGNN, and TGF-M remain 3D or
teacher protocols; and Edge-Set, EquiHGNN, GoMS, and nonstandard 3D PCQM
numbers are not accepted as comparable scores.

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
