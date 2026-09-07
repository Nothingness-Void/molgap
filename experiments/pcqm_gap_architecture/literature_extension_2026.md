# Incremental Literature Extension for PCQM Gap (2026-09-07)

This document extends the original 73-source literature ledger without
rewriting its historical rows. The synchronized coverage ledger now contains
195 unique audited sources; rows 1--73 remain the original base authority and
the latest one hundred and twenty-two sources are also synchronized as rows 74--195:
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

### E16j. NDI linker gap screening: descriptor-only surrogate with a narrow chemical domain

**Primary source.** [Achour et al., Materials Chemistry and Physics 2026](https://doi.org/10.1016/j.matchemphys.2026.132345), with the [open UCL record and published-version PDF](https://discovery.ucl.ac.uk/id/eprint/10223000/).

This study builds a focused library of `195` naphthalene-diimide (NDI)
derivatives whose substituents vary around a fixed pi-conjugated core. Gaussian
16 B3LYP/6-31G(d,p) calculations provide the reference HOMO--LUMO gaps; the
ML inputs are RDKit descriptors calculated from SMILES rather than DFT orbitals
or 3D coordinates. The abstract reports LightGBM as the best model with
`R^2 = 0.86` and `RMSE = 0.25 eV`; the full article also exposes a test-set
figure around `R^2 = 0.69`, `RMSE = 0.37 eV` and a bootstrap summary around
`R^2 ~= 0.82`, `RMSE ~= 0.29 eV`. These are not interchangeable numbers, so
the source should be cited with the exact evaluation table/figure rather than
as one universal score.

The authors additionally compare 32 literature NDI compounds, separating
optical (`n = 24`) and electrochemical (`n = 8`) gaps from the Kohn--Sham
quantity. On that heterogeneous external set, the LightGBM surrogate reports
`MAE = 0.650 eV`, `RMSE = 0.916 eV`, and positive bias `+0.516 eV`; the paper
explicitly warns that optical and redox gaps include effects not represented by
a gas-phase Kohn--Sham HOMO--LUMO difference. The reported B3LYP calculation
cost is `761.24` CPU-hours for the 195 labels, with a claimed screening
break-even near 195 candidates.

**MolGap reading.** This is useful evidence for a cheap descriptor baseline,
scaffold-restricted applicability-domain reporting, SHAP interpretation, and
explicit separation of DFT labels from experimental gap definitions. It does
not establish a PCQM4Mv2 result: the fixed NDI scaffold is a narrow domain, the
basis is `6-31G(d,p)` rather than the project's stated `6-31G*`, the DFT
geometry contract is not ETKDG, and no official code/checkpoint or general
chemical-space split was found in the primary release surface. The external
experimental comparison must not be relabeled as B3LYP/6-31G* evidence.

**Disposition.** B for organic-electronics protocol and descriptor/uncertainty
reference; C for current labels, database, initialization, and experiment.

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

### E31. NVIDIA PCQM4Mv2 winner: heterogeneous ensemble and OOF stacking

**Primary sources.** [Heterogenous Ensemble of Models for Molecular Property
Prediction](https://arxiv.org/abs/2211.11035) and the authors' [MIT-licensed
NVIDIA-PCQM4Mv2 repository](https://github.com/jfpuget/NVIDIA-PCQM4Mv2).

This completed 2022 OGB-LSC solution reports a test-challenge MAE of `0.0723`,
a validation MAE of `0.07145`, and under two hours of inference. It releases
the fold builder, Transformer-M/Graphormer, molecular Transformer, PD-DGN, CNN,
and ensemble directories. Its key reproducible idea is not merely ensembling:
the authors diagnose a heavy-atom distribution shift (train up to 20 versus
validation/test-dev/test-challenge up to 51), create 24 folds from combined
train+valid rows, train four held-out-fold families, store OOF predictions, and
fit a HuberRegressor on those predictions. The paper reports 39 checkpoints
from 10 models.

The base ablations also separate two auxiliary ideas: Transformer-M position
denoising and KPGT-style fingerprint/descriptor regularization. The released
configuration documents an 18-layer width-768 model, position noise `0.2`,
batch `128`, 454 epochs, and eight distributed processes; the paper reports an
initial `0.007` improvement for the descriptor/fingerprint regularizer. The
published stack gives the denoising model a small negative coefficient while
retaining positive weights for several weaker models, showing that ensemble
diversity and single-model quality need not agree.

**MolGap disposition.** A for completed direct-PCQM code/evidence and B for
current transfer. Borrow only the size-shift diagnostic, OOF manifest, and
Huber-stacking protocol after the candidate list and sealed evaluation roles
are frozen. The solution uses PCQM-provided 3D SDF geometry, RDKit image views,
train+valid fold construction, and large multi-checkpoint compute, so its
`0.0723` cannot be used as a current ETKDG/GraphState result. No rows,
checkpoints, folds, or image artifacts are imported. See the [full public-code
deep-reading card](deep_reading_public_code_2026-09-07.md#54-nvidia-pcqm4mv2-winner-completed-heterogeneous-ensemble-and-oof-stacking).

### E32. OSCAgent: graph--SMILES alignment plus auxiliary LUMO supervision

**Primary sources.** [OSCAgent: Accelerating the Discovery of Organic Solar
Cells with LLM Agents](https://arxiv.org/html/2602.04510) and the public
[Harvard Clean Energy Project data record](https://www.nature.com/articles/sdata201686).

OSCAgent is adjacent organic-photovoltaic evidence rather than a direct PCQM
result. Its computational pretraining role contains `51,256` molecules from
the Lopez/Harvard screening lineage, and its downstream role is an experimental
OSC data set curated by Sun et al. The model aligns graph and SMILES encoders
with a symmetric InfoNCE loss, adds LUMO regression to both branches, and then
fuses the two embeddings with Morgan/MACCS fingerprints in a mixture-of-experts
PCE head. The paper reports full-model PCE `R^2/MAE = 0.713/1.686`, compared
with `0.654/1.879` without pretraining; removing graph, SMILES, handcrafted
features, or the uncertainty loss gives respectively `0.665/1.822`,
`0.675/1.793`, `0.634/1.924`, and `0.681/1.776`.

The reliable conclusion is an objective pattern: same-identity multi-view
alignment and an electronic auxiliary head can be tested separately, with
uncertainty as a calibration control. The numbers cannot be transferred to
MolGap because the reported endpoint is experimental PCE, the pretraining
theory/geometry contract is not closed to B3LYP/6-31G*/ETKDG, and no official
code, checkpoint, or independently retrievable pretraining release was found.
The computational and experimental rows must not enter Track A/B. **Grade:
C for artifacts/direct use; B for a future objective reference.** See the
[full organic-electronics deep-reading card](deep_reading_organic_electronics_2026-09-07.md#11-oscagent-graphsmiles-contrastive-pretraining-with-frontier-supervision).

### E33. QuADMET-Former: public quantum-target bundle and E3FP control

**Primary sources.** [QuADMET-Former preprint](https://doi.org/10.26434/chemrxiv.15002429/v1),
the [MIT-licensed code](https://github.com/arunraja-hub/quadmetformer), its
[Hugging Face model card and checkpoint surface](https://huggingface.co/arunraja007/quadmetformer),
and the exact public [QMugs pretraining configuration](https://raw.githubusercontent.com/arunraja-hub/quadmetformer/main/pretraining/configs/pretraining/qmugs.json).

This is a completed public pretraining asset, but not a current PCQM result.
The configuration names `TorchMD_ET` with atom-level Löwdin charge supervision
and molecular targets `E3FP`, dipole moment, HOMO--LUMO Gap, and total energy.
The project README/model card reports 23 ADMET downstream benchmarks and a best
result on `16/23` tasks; the searchable preprint description reports that the
quantum properties include HOMO/LUMO/Gap, dipole, partial charges, and
TPSSh-D3BJ/def2-SVP-related QMugs calculations. The public loss code adds a
molecular formal-charge conservation term to the atom-charge loss.

The artifact audit also verified the GitHub `main` revision
`b5789c52a2701009005b98930fae24f8c136142f` and Hugging Face revision
`a106f8a675d44b4291ec6977666cf12cc12fa912`; the model listing exposes named
single- and multi-conformer `.ckpt` files. This is sufficient to grade the
source as a real public code/weight package, but not to close the PCQM identity,
geometry, or target-theory contract.

The reliable contribution is the control design: separate local electronic,
global frontier, global dipole/energy, and geometry-fingerprint targets, then
measure the target bundle rather than assuming every auxiliary task helps. Its
limitations are equally clear: QMugs conformers and theory are external, E3FP
depends on a non-frozen 3D input role, ADMET is the downstream endpoint, and no
same-contract PCQM Gap number is reported. **Grade: A/B for code/configuration
and teacher-loss design; C for current labels, geometry, weights, or direct
experiment.** See the [full electronic/delta deep-reading card](deep_reading_electronic_delta_2026-09-07.md#721-quadmet-former-a-complete-public-quantum-property-pretraining-asset).

### E34. BOA: basis-overlap message passing for electron-density teachers

**Primary sources.** The ICLR 2026 [BOA paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/718573eff1cb169316783d3e08514b5b-Paper-Conference.pdf), the official [LGPL-3.0 code](https://github.com/sciai-lab/boa), its [raw README](https://raw.githubusercontent.com/sciai-lab/boa/main/README.md), and the public [Hugging Face checkpoint page](https://huggingface.co/sciai-lab/boa).

BOA is a completed, auditable electron-density architecture rather than a
direct HOMO/LUMO/Gap model. It expands the density in atom-centred Gaussian
basis products, represents the resulting density-matrix blocks without
materialising a full matrix, and uses overlap-integral projections for
equivariant message passing and attention. The paper reports QM9 density-grid
NMAE of `0.1381 +/- 0.0003` (VASP, small), `0.13 +/- 0.01` (PySCF, small),
`0.1339 +/- 0.0005` (VASP, large), and `0.116 +/- 0.006` (PySCF, large); the
same paper also reports stable size transfer from QM9 to QMugs molecules up to
nearly 200 atoms. The repository exposes loaders, training/testing entry points,
and a `qm9_pyscf_large.ckpt` example surface.

The transferable idea is a physics-grounded teacher representation: basis
overlap can provide richer local electronic structure than a scalar Gap label,
while the density target remains spatially decomposable. The evidence does not
authorize a MolGap run: BOA predicts electron density, not the current
B3LYP/6-31G* Kohn--Sham HOMO/LUMO/Gap; its VASP/PySCF QM9 and QMugs density
datasets use external geometry/theory contracts; and no same-contract PCQM
checkpoint or Gap score is reported. **Grade: A/B for public density-teacher
architecture and code; C for current labels, weights, geometry, or direct
experiment.** See the [full electronic/delta deep-reading card](deep_reading_electronic_delta_2026-09-07.md#722-boa-basis-overlap-message-passing-for-electron-density-teachers).

### E35. C-FREE: contrast-free ego-net prediction with public 3D checkpoints

**Primary sources.** The ICML 2026 [C-FREE paper](https://arxiv.org/html/2509.22468),
the authors' [MIT implementation](https://github.com/ariguiba/C-FREE), and the
public [Hugging Face checkpoints](https://huggingface.co/ariguiba/C-FREE).

C-FREE is a non-contrastive latent-predictive objective. A sampled node defines
a fixed-radius ego-net and complementary subgraph; a context encoder plus
predictor matches the EMA target encoder embedding. The views can be 2D, 3D,
or multimodal, and the paper reports predictor removal collapse. Pretraining
uses about `304,466` GEOM molecules and about `25M` conformers; the public
repository/checkpoint surface was frozen in the full card. Its grouped QM9
`HOMO/LUMO/GAP` column improves over random PaiNN, but remains behind Uni-Mol2
and is not a PCQM4Mv2 Gap result. **Grade: A/B for objective, code, checkpoint
provenance, and controlled QM9 evidence; C for current PCQM/ETKDG use.** See
the [full geometry/higher-order deep-reading card](deep_reading_foundation_geometry_higher_order_2026-09-07.md#15-c-free-contrast-free-ego-net-prediction-with-23d-conformers).

### E36. OneQMC/Orbformer: public wavefunction and density-teacher assets

**Primary sources.** The [Orbformer paper](https://arxiv.org/abs/2506.19960),
the [MIT OneQMC repository](https://github.com/microsoft/oneqmc), its [model
card](https://github.com/microsoft/oneqmc/blob/main/model_card.md), and the
companion [NERD electron-density paper](https://arxiv.org/abs/2409.01306).

Orbformer is a transferable variational neural wavefunction pretrained on
`22,350` Light Atom Curriculum configurations, with public code, data,
notebooks, and an `lac.chkpt` checkpoint. The paper reports chemical-accuracy
bond-breaking/Diels--Alder evidence and large cost reductions from pretraining
and joint geometry-curve fine-tuning; NERD adds a physics-constrained
real-space density extraction route. This is a credible electronic-teacher
asset, but it is not a scalar HOMO/LUMO/Gap predictor. The public model card
also warns that zero-shot use is insufficient, supports only H/Li/B/C/N/O/F in
the released checkpoint, and assumes high-memory JAX/A100 execution. **Grade:
A/B for teacher concept, code, checkpoint, and controlled evidence; C for
current PCQM/ETKDG labels, weights, geometry, and direct experiment.** See the
[full electronic/delta deep-reading card](deep_reading_electronic_delta_2026-09-07.md#723-oneqmcorbformer-a-public-wavefunction-teacher-not-a-gap-regressor).

### E37. OrbitAll: orbital-feature delta learning with a missing code release

**Primary source.** The [OrbitAll paper](https://arxiv.org/html/2507.03853).

OrbitAll is a highly relevant method-level result: spin-polarized Fock/density/
overlap/core-Hamiltonian features from spGFN1-xTB or g-xTB are fed to an
SE(3)-equivariant GNN that predicts `y_target - y_low-level`. On QM9star
(`B3LYP-D3(BJ)/6-311+G(d,p)`), the paper reports direct-vs-delta and
spin-resolved HOMO/LUMO controls, including OrbitAll(`Delta`) total-energy MAE
`8.50 meV` versus `14.75 meV` for direct OrbitAll and radical alpha/beta
frontier-level errors of `55.40/43.31/32.27/14.73 meV`. It also reports roughly
`7K` versus `70K` training examples to reach chemical accuracy in its learning
curve. However, the audited paper only promises code/data release upon
publication; no verified official implementation or checkpoint was found.
**Grade: B for delta/orbital-feature evidence; C for current code, weights,
geometry, labels, and direct PCQM experiment.** See the [full
electronic/delta deep-reading card](deep_reading_electronic_delta_2026-09-07.md#724-orbitall-orbital-features-plus-explicit-delta-learning).

### E38. NN-xTB: Hamiltonian-level low-fidelity teacher with Code Ocean artifacts

**Primary sources.** The [Nature Communications paper](https://www.nature.com/articles/s41467-026-73184-z),
its [Code Ocean capsule](https://doi.org/10.24433/CO.8668201.v1), and the
[release-status repository](https://github.com/Barca-group/NN-xTB).

NN-xTB predicts bounded environment-dependent shifts to GFN2-xTB Hamiltonian
parameters and solves the modified SCF problem, preserving charge/spin and
electronic-state consistency. The paper reports GMTKN55 WTMAD-2 `3.78` versus
`25.0` for GFN2-xTB, a VQM24 frequency MAE of `12.7 cm^-1` versus `200.6`, and
less than `20%` runtime overhead, while the paper explicitly provides source,
pretrained-model, and script access through Code Ocean. It has no verified
same-contract HOMO/LUMO/Gap result; its role is a low-fidelity Hamiltonian
teacher/delta reference, not a current experiment. **Grade: A/B for method and
artifact provenance; C for current frontier labels, geometry, and direct use.**
See the [full electronic/delta deep-reading card](deep_reading_electronic_delta_2026-09-07.md#725-nn-xtb-hamiltonian-level-low-fidelity-teacher-with-a-reproducible-archive).

### E39. Message-Passing Delta-ML: public organic-electronics residual workflow

**Primary sources.** The peer-reviewed [JCTC paper](https://doi.org/10.1021/acs.jctc.5c01587)
and its [public code/data/model repository](https://github.com/AdamCoxson/Message-Passing-Delta-ML).

This workflow predicts M06-2X/3-21G* TDDFT `S1` energies from ZINDO by adding a
message-passing residual to the low-level `S1` value, with Mulliken, electron/
hole, orbital, and MO-weighted radial descriptors. It reports `0.964` test
correlation for the best S1 model and `0.839` correlation for an adapted
oscillator-strength model, with public pretrained models and tutorials. The
conjugated-core split and explicit low-level residual are strong protocol
evidence for organic electronics, but `S1` is not the current Kohn--Sham Gap
target and the geometry/theory roles are external. **Grade: A/B for public
Delta workflow and optical evidence; C for current PCQM/ETKDG use.** See the
[full organic-electronics deep-reading card](deep_reading_organic_electronics_2026-09-07.md#12-message-passing-delta-ml-for-excited-state-organic-electronics).

### E40. Ordered-to-disordered transfer: HOMO--LUMO gaps expose chemistry-shift failure

**Primary source.** [Untarabut et al., arXiv:2607.29510](https://arxiv.org/abs/2607.29510).

This is an adjacent materials result rather than a molecular PCQM benchmark,
but it is a clean transfer-learning warning. The authors train CGCNN, GATGNN,
ALIGNN, and M3GNet on chemically ordered single/double perovskites and test
transfer to chemically disordered high-entropy perovskite oxides (HEPOs).
ALIGNN, the angle-aware model, reaches ordered-domain HOMO--LUMO-gap test MAE
`0.03 eV`, versus approximately `0.04` for CGCNN/GATGNN and `0.05` for M3GNet.
In the strict blind transfer, the same ordered-only ALIGNN gives `0.28 eV`
HEPO Gap MAE while formation-energy MAE remains `1.98 meV/atom` versus
`2.47 meV/atom` on ordered structures. The authors attribute the electronic
failure to chemically disordered local environments and systematic
underprediction; small HEPO-specific additions then improve the Gap, but the
fine-tuning curve is not a blind test because HEPO validation is used for model
selection.

**MolGap reading.** The result supports two bounded conclusions. First,
angular/three-body representations can help when local orbital overlap and
distortion are genuinely encoded; this is corroborative, not a license to
reopen the already tested torsion/contact mechanisms. Second, a good in-domain
Gap score does not demonstrate transfer across a chemistry shift. For MolGap,
the analogous control is to report scaffold/size/electronic-family residuals
and to keep a teacher or pretraining corpus identity-disjoint, even when its
global target distribution looks similar. The source uses periodic solid-state
structures and DFT data, not molecules, and it supplies no current ETKDG or
PCQM artifact; it is therefore a negative/OOD protocol reference only.

**Disposition.** B for chemistry-shift and angle-aware transfer analysis; C for
current model selection, labels, geometry, and database use.

### E41. QuantumCanvas: a public two-body electronic teacher asset

**Primary sources.** The [arXiv paper](https://arxiv.org/abs/2512.01519), [MIT code](https://github.com/KurbanIntelligenceLab/QuantumCanvas), and [CC-BY Zenodo dataset](https://doi.org/10.5281/zenodo.20631934) are independently retrievable. The paper defines `2,850` element-pair diatomics across `75` elements, ten-channel orbital/charge image representations, and frontier, energy, charge, and geometry labels. The repository exposes `dataset_combined.npz`, an MD5, loaders, and `REPRODUCE.md`.

The paper compares eight architectures under element-pair-disjoint splits and three seeds. For the gap task it reports GATv2 at `0.201 +/- 0.020 eV`, EGNN at `0.226 +/- 0.015 eV`, and DimeNet at `0.248 +/- 0.020 eV`; pretraining is then transferred to QM9, MD17, and CrysMTM. The QM9 fine-tuning setup is `110K/10K/10K`, with scratch learning rate `1e-4` and fine-tuning learning rate `1e-5`.

**Contract audit.** The paper introduces a finite-temperature Kohn--Sham/Mermin formulation, but its dataset section and repository identify the actual calculation as SCC-DFTB/DFTB+ with PTBP parameters. It is therefore not a B3LYP/6-31G* label source. It contains isolated two-body systems with explicit/DFTB-derived coordinates, not molecular PCQM records or ETKDG conformers. The paper's target count (`18`, with a separate detailed-materials count of `22`) also does not match the repository README's `37` label keys; the exact revision/schema must be frozen before reuse. No direct PCQM Gap result or current-contract checkpoint is exposed.

**Disposition.** B for two-body/electronic-teacher design, composition-held-out
evaluation, and artifact-manifest discipline; C for current labels, database,
geometry, weights, and experiment. Do not import its rows, image tensors, or
weights.

### E42. Public HOMO/LUMO databases with distinct audit roles

These sources are useful because they expose completed, independently
retrievable electronic data, but none is a silent extension of the current
database.

- The [DTU solar database](https://cmr.fysik.dtu.dk/solar/solar.html) explicitly
  exposes B3LYP Kohn--Sham `KS_gap`, `E_homo`, and `E_lumo` fields, together with
  optical-gap, tight-binding, dimer-shift, dipole, and `V_oc` fields for organic
  donor--acceptor molecules. The primary page does not establish the basis,
  record count, coordinate construction, or redistribution license. It is a
  narrow organic-electronics family/OOD audit source, not a merge candidate.
- The [tmQM paper](https://pubs.acs.org/doi/10.1021/acs.jcim.0c01041) and [2024 public repository](https://github.com/uiocompcat/tmqm) provide the original `86,665` and expanded roughly `108K` transition-metal complexes. The repository exposes GFN2-xTB geometries and TPSSh-D3BJ/def2-SVP HOMO/LUMO/Gap, dipoles, metal charges, and related fields. It is a strong metal/electronic OOD or teacher reference, but not the current organic B3LYP/6-31G*/ETKDG contract.
- [BOS-TMC](https://doi.org/10.1021/acs.jcim.6c01792) provides `159,014` experimentally characterized complexes and up to `343.8K` complex/spin combinations, with PBE0/def2-TZVP single points on crystal-derived heavy-atom coordinates and optimized hydrogens. It exposes HOMO/LUMO/Gap, charges, dipoles, atomization, and spin-splitting properties. Its open-shell, broad-charge, transition-metal, and experimental-geometry roles make it valuable for OOD/electronic teacher audits, but they are radically mismatched to the current target and geometry contract.

**Disposition.** B for data-contract, theory-shift, charge/spin, and domain-shift
audits; C for row merge, current labels, and direct training. Keep the current
PCQM/repaired-2M databases unchanged.

### E43. TMC-Delta-ML: completed public fidelity and residual-learning protocol

**Primary sources.** The peer-reviewed [Chemistry--A European Journal paper](https://doi.org/10.1002/chem.71487), [MIT TMC-Delta-ML code](https://github.com/uiocompcat/TMC-Delta-ML), [tmQMg data/code](https://github.com/uiocompcat/tmQMg), and [Zenodo low-fidelity graph archive](https://doi.org/10.5281/zenodo.18348669) form a complete enough source chain for method audit. The repository is MIT-licensed and exposes `bench`, `lsda`, and `pbe0` modes, fixed CSD-ID split files, and automatic tmQMg data retrieval.

The method uses GFN2-xTB geometries and low-fidelity u-NatQG graph features to
predict tmQMg high-fidelity properties. The public code explicitly supports
HOMO--LUMO Gap, polarizability, and dipole without the atomic-energy fitting
step, and it compares a direct benchmark with low-cost GFN2-xTB//LSDA/LANL2DZ
and GFN2-xTB//PBE0-D3BJ/def2-TZVP fidelity modes. The paper reports improved
accuracy, data efficiency, and out-of-domain transfer over its benchmark, while
showing that a cheaper low-fidelity route can trade a small amount of accuracy
for a substantial cost reduction.

**MolGap reading.** This is stronger Δ-learning evidence than a method-only
claim because the code contains the benchmark/residual switch, low-fidelity
graph artifacts, fixed identity splits, and a target-specific HOMO--LUMO path.
It still uses transition-metal/CSD chemistry, u-NatQG features, GFN2-xTB
geometry, and PBE0-D3BJ/def2-TZVP or LSDA/LANL2DZ roles. It does not establish a
residual route for PCQM4Mv2's B3LYP/6-31G* labels or the ETKDG inference path.

**Disposition.** A/B for Δ-learning protocol, fidelity/cost controls, and public
artifact packaging; C for current labels, database, geometry, and direct
experiment. Borrow only the benchmark-versus-residual and cheap-versus-
expensive fidelity matrix after a separately authorized same-PCQM audit.

### E44. Selected Machine Learning: chemical-family stratification before Δ-ML

**Primary sources.** The peer-reviewed [Materials Advances paper](https://doi.org/10.1039/D2MA00742H), [arXiv record](https://arxiv.org/abs/2110.02596), and [public SelectedML repository](https://github.com/b3rn4rdm/SelectedML) provide both the method and executable analysis scripts. The code performs frequency analysis from molecular structure, writes class-labelled representations, runs separate direct QML learning curves, and includes a QM7b direct/Delta script.

The paper partitions QM7/QM9 molecules into three chemically interpretable
classes: aromatic-ring/carbonyl systems, singly unsaturated systems, and
saturated systems. It evaluates kernel-ridge models with CM, BoB, and SLATM
representations at GW, B3LYP, and ZINDO levels. The primary result is a lower
learning-curve offset after class selection: the paper reports approximately
`0.1 eV` MAE with up to an order-of-magnitude fewer training molecules, and
the QM9 saturated class reaches `0.1 eV` with about `16K` examples versus more
than `64K` for an unselected pool. Its direct-versus-Delta comparison is useful
because it tests whether reducing chemical heterogeneity can beat a low-level
residual target under small-data conditions.

**MolGap reading.** This is evidence for stratified error analysis and a
possible mixture-of-experts or class-conditioned readout, not evidence for
splitting the official PCQM4Mv2 training set into independently trained models.
The source datasets, KRR/Coulomb-like representations, theory levels, and
coordinate contracts differ from B3LYP/6-31G*/ETKDG PCQM. Class rules must also
be defined from input chemistry only, with all official validation/test roles
kept sealed; a post-hoc class-specific metric is safer than a new model route.

**Disposition.** A/B for chemical-family stratification, learning-curve, and
direct-versus-Delta control design; C for current weights, labels, geometry,
and direct experiment. If revisited, borrow the class definitions and report
family-conditioned residuals before considering a separately authorized
mixture-of-experts screen.

### E45. HLP-Stack: public descriptor ensemble with a target-adjacent feature warning

**Primary sources.** The open-access [RSC Advances paper](https://doi.org/10.1039/D5RA08007J), [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959570/), and [public HLP-STACK repository](https://github.com/college-of-pharmacy-gachon-university/HLP_STACK) expose the paper, raw/processed data, notebooks, saved models, figures, and environment file.

HLP-Stack combines RDKit 2D descriptors with 3D quantum descriptors from QM9,
starts from `221` descriptors, selects a final `51`, and stacks Random Forest,
XGBoost, Extra Trees, and Gradient Boosting under a linear meta-learner. The
paper reports test RMSE of `3.219e-4 Eh` for HOMO and `1.903e-4 Eh` for LUMO,
with (R^2 approx 0.9999), and the repository contains the corresponding
model/data artifact surface.

**Contract and evidence warning.** QM9 is B3LYP/6-31G(2df,p), not PCQM4Mv2
B3LYP/6-31G*. More importantly, the 3D feature path uses QM9 quantum
descriptors calculated at the DFT level, including target-adjacent electronic
and geometric quantities; those are not available from the current SMILES/
ETKDG inference contract without an extra quantum calculation. The near-perfect
metrics and the repository's perfect-looking training/validation summaries
therefore require a feature-by-feature provenance and split audit before they
can be interpreted as a model result. They are not comparable to the current
GraphState screen.

**Disposition.** B for feature-provenance, leakage, and descriptor-baseline
audit; C for current experiment. Keep it as a negative control: a future
descriptor baseline may use only descriptors computable from the same ETKDG
graph/coordinates before labels are read, and must be compared with a strict
SMILES/ETKDG control. Do not import HLP-Stack models or QM9 quantum features.

### E46. PET-MAD-DOS: complete DOS/band-gap teacher artifact

**Primary sources.** The [Digital Discovery paper](https://doi.org/10.1039/D5DD00557D),
[arXiv record](https://arxiv.org/abs/2508.17418), [Materials Cloud reproduction
record](https://doi.org/10.24435/materialscloud:gs-z7), [UPET code](https://github.com/lab-cosmo/upet),
and [Hugging Face model](https://huggingface.co/lab-cosmo/pet-mad-dos) jointly
establish the artifact. PET-MAD-DOS is a PET Transformer trained on MAD to
predict a fixed-grid smoothed DOS. The release carries `gap`, DOS, reliability
`mask`, electron count, data-preparation, training, fine-tuning, LoRA, UQ, and
gap-extraction scripts.

The paper demonstrates external-sample evaluation and low-data fine-tuning,
while the reproduction record makes the data and scripts retrievable. This is
useful evidence for an electronic teacher whose output is a structured object,
not only one scalar. However, the data distribution and labels are
heterogeneous materials/molecule PBEsol/Quantum Espresso calculations; the
DOS-derived `gap` is not a B3LYP/6-31G* PCQM HOMO/LUMO pair, and the geometry
is not the current ETKDG cache.

**Disposition.** A/B for completed electronic-teacher packaging, DOS/UQ/LoRA
design, and artifact-manifest lessons; C for current rows, weights,
initialization, and experiment. Do not import its data or equate its gap with
the MolGap target.

### E47. FieldMACE: multipole-derived long-range messages with public code/data

**Primary sources.** The [npj Computational Materials paper](https://doi.org/10.1038/s41524-026-02048-3),
[official repository](https://github.com/rhyan10/FieldMACE), and [Figshare
archive](https://figshare.com/articles/dataset/Models_data_and_code_for_publication_Incorporating_Long-Range_Interactions_via_the_Multipole_Expansion_into_Ground_and_Excited-State_Molecular_Simulations_/28497857)
are available. FieldMACE adds multipole-derived long-range fields to MACE
messages for QM/MM, uses direct summation to retain low scaling, and studies
higher-order multipoles, foundation-model transfer, solvated molecules,
nickel-complex dynamics, and excited states. The paper/code expose a concrete
5 Å local graph, `ell` truncation, `--foundation_model`, H100 training, and
energy/force loss contract.

The transferable idea is where the long-range information enters: at the
message level, so the environment can affect many-body interactions, rather
than as an independent post-hoc scalar correction. The paper also documents a
short-range QM/MM boundary failure and the cost of increasing multipole order.
This is not a PCQM Gap result: it requires explicit MM point charges and has
energy/force rather than HOMO/LUMO/Gap targets.

**Disposition.** A/B for long-range message and foundation-transfer design; C
for current target, database, ETKDG geometry, and experiment. No multipole or
MM feature generation is admitted.

### E48. MACE-POLAR-1: charge/spin-constrained electrostatic foundation model

**Primary sources.** The [paper](https://arxiv.org/html/2602.19411), [official
foundation release](https://github.com/ACEsuit/mace-foundations/releases), and
[MACE-POLAR documentation](https://mace-docs.readthedocs.io/en/latest/guide/polar_mace.html)
provide method and checkpoint evidence. MACE-POLAR-1 extends MACE with a
non-self-consistent polarizable field, Gaussian-smeared multipoles, repeated
long-range updates, and global Fukui equilibration for total charge and spin.
It is trained on 100M OMol25 hybrid-DFT structures and is evaluated on
thermochemistry, conformers, non-covalent interactions, transition metals,
charge transfer, and external-field response.

The valuable mechanism is a constrained electronic representation: local
features generate charge/spin multipoles, non-local fields update them, and
global normalization enforces system-level charge/spin. The released models
are not frontier-orbital predictors and have no direct PCQM4Mv2 result. Their
`omegaB97M-V`/OMol25 3D energy-force contract is external to
B3LYP/6-31G*/ETKDG.

**Disposition.** A/B for electrostatic foundation and physically constrained
teacher design; C for current initialization, labels, database, and experiment.
OMol25 rows and released weights remain external.

### E49. MACE-H: electronic-operator prediction with high-body-order messages

**Primary sources.** The updated [MACE-H paper](https://arxiv.org/html/2508.15108),
[MIT code](https://github.com/maurergroup/MACE-H), and [Zenodo reproduction
package](https://doi.org/10.5281/zenodo.15223696) provide paper, code, configs,
containers, and post-processing. MACE-H predicts local Kohn--Sham Hamiltonian
blocks in an atomic-orbital basis. High-body-order messages and a node-degree
expansion supply the angular-momentum content needed for orbital blocks; Julia
tools diagonalize predicted reciprocal-space matrices into bands and DOS. The
reported benchmarks are periodic 2D materials and bulk Au, with sub-meV matrix
errors and downstream eigenvalue/DOS analysis.

This is a strong completed example of operator-level electronic supervision and
of analyzing body order versus locality, but it is not a molecular PCQM Gap
system. OpenMX/FHI-aims basis conventions, SOC, periodic cells, and reciprocal
space are all external to the current B3LYP/6-31G*/ETKDG contract.

**Disposition.** A/B for operator-level/high-body-order method reading and
reproducible packaging; C for MolGap weights, data, geometry, and experiment.
Do not duplicate its materials score as a PCQM molecular claim.

### E50. CheMeleon: descriptor-pretrained D-MPNN with released data and weights

**Primary sources.** The [CheMeleon paper](https://arxiv.org/html/2506.15792),
[official repository](https://github.com/JacksonBurns/chemeleon), [training-data
release](https://doi.org/10.5281/zenodo.15733574), [model-weight
release](https://doi.org/10.5281/zenodo.15426600), and [Chemprop fine-tuning
documentation](https://chemprop.readthedocs.io/en/main/chemeleon_foundation_finetuning.html)
jointly establish a usable molecular foundation-model artifact. The published
model is a roughly 10M-parameter six-layer D-MPNN/FNN trained on one million
PubChem molecules to predict Mordred descriptor bundles. It uses dynamic 85%
descriptor masking; for a downstream task, the descriptor FNN is discarded, a
new task head is attached, and the pretrained D-MPNN is fine-tuned end to end.

The evidence is strong for a **pretraining mechanism and engineering route**,
not for a current PCQM4Mv2 Gap result. The published evaluation covers 58
MoleculeACE/Polaris tasks, but no matched B3LYP/6-31G* PCQM4Mv2 HOMO/LUMO/Gap
score is exposed in the verified materials. PubChem is external to the current
database, the input is graph/SMILES rather than ETKDG coordinates, and the
descriptor bundle is not the current Kohn--Sham target. The descriptor list
therefore needs a source and target-adjacency audit before any same-database
reuse. The repository also reports a full reproduction cost of about 500 CPU
hours for descriptor calculation, 1000 GPU-hours for pretraining, and over 1 TB
of checkpoint/storage footprint; this is outside the current bounded screen.

The safe future adaptation, if separately authorized, is to compute an
explicitly frozen descriptor subset from **PCQM training molecules only**, keep
validation/test molecules out of pretraining, attach a fresh Gap head, and
compare against a paired random-init control under the unchanged ETKDG
inference route. No external PubChem rows, CheMeleon weights, or current run
are admitted.

**Disposition.** A/B for descriptor-pretraining design, source packaging,
checkpoint/data handling, and cost planning; C for the external foundation
route, weights, database expansion, and current experiment.

### E51. Zatom-1: released 3D flow foundation with frontier-property heads

**Primary sources.** The [Zatom-1 paper](https://arxiv.org/html/2602.22251),
[official repository](https://github.com/Zatom-AI/zatom), and [Zenodo checkpoint
record](https://zenodo.org/records/19766997) provide a complete public artifact.
Zatom-1 uses a Trunk-based Flow Transformer: multimodal flow pretraining over
atom types and explicit 3D coordinates (plus material lattice modalities),
followed by predictive property/energy/force heads. The official QM9
evaluation command includes `homo`, `lumo`, and `gap`, and the release includes
generative, QM9-only, joint, non-pretrained, and property-prediction
checkpoints. The main model is about 77M parameters with 160M/300M variants.

The paper provides useful controls: full-trunk unfreezing harms generative
validity, while trunk freezing and LoRA are distinct transfer routes. It also
reports about 20,000 GPU-hours and 1 TB storage across the project. This is
strong teacher/checkpoint evidence but not current PCQM evidence: properties
are QM9/Matbench, energy/force roles are OMol25/MPtrj, explicit 3D inputs are
used, and no matched PCQM4Mv2 B3LYP/6-31G* Gap or ETKDG route is published.

**Disposition.** A/B for 3D foundation/pretraining design, frozen-trunk versus
LoRA controls, and artifact packaging; C for current weights, external rows,
coordinates, database, and experiment. No Zatom asset is admitted.

### E52. OrbNet-Equi: orbital-basis electronic features with direct/Delta crossover

**Primary sources.** The [PNAS paper](https://doi.org/10.1073/pnas.2205221119),
[arXiv record](https://arxiv.org/abs/2105.14655), and [Zenodo source/data/code
package](https://zenodo.org/records/6568437) provide an auditable electronic
Delta-learning asset. OrbNet-Equi runs a low-cost GFN-xTB mean-field
calculation, converts its electronic operators to an atomic-orbital feature
stack, and applies the equivariant UNiTE network. On QM9 it compares direct
learning with residual learning for frontier orbital properties. The paper's
default-feature learning curves show a Delta advantage that shrinks with data;
the LUMO and Gap curves cross at roughly `32K--64K` training examples, while
energy-weighted occupied/virtual density matrices preserve a Delta/direct gap.

This is a method result, not current PCQM evidence. QM9 uses its standard DFT
labels; the separately assembled SDC21 route uses non-equilibrium geometries
and `omegaB97X-D3/def2-TZVP` labels. GFN-xTB operator features, QM9/SDC21
identities, and the absence of a matched PCQM4Mv2 B3LYP/6-31G*/ETKDG result
block direct import. The Zenodo record nevertheless exposes source data,
examples, code, and checksums, making the artifact stronger than a paper-only
Delta claim.

**Disposition.** A/B for the direct-versus-scalar-residual-versus-orbital-
residual control design and electronic-feature diagnosis; C for GFN-xTB
features, external rows/geometries, weights, and current initialization.

### E53. Image-super-resolution electron density: a real-space electronic teacher

**Primary sources.** The [Nature Communications paper](https://doi.org/10.1038/s41467-025-60095-8),
[pretrained-model/code record](https://doi.org/10.5281/zenodo.15226766), and
[CC-BY Figshare data release](https://figshare.com/articles/dataset/Image_Super-resolution_Inspired_Electron_Density_Prediction/25365508)
provide a public density-teacher package. A 3D convolutional ResNet maps a
coarse superposition of neutral atomic densities to a high-resolution
ground-state density. The QM9 density labels use PBE, GTH pseudopotentials,
and a GTH-TZV2P basis in a Gaussian--Plane-Wave/PySCF workflow. A one-step
Kohn--Sham Fock build and diagonalization then yields energy, HOMO, LUMO, and
Gap; the reported QM9 one-step Gap error is about `11 meV`.

The result is not a pure learned frontier-property head: the diagonalization
is part of the evaluation path. The paper also reports transfer to
non-equilibrium conformers and limited-data fine-tuning for unseen elements
and charge states. Real-space grids, PBE/GTH density labels, QM9/water/MD
roles, and no PCQM4Mv2 B3LYP/6-31G*/ETKDG route keep it external to MolGap.

**Disposition.** A/B for electron-number-normalized density teaching,
geometry-transfer controls, and the distinction between density error and
post-diagonalization orbital error; C for grid data, PBE labels, pretrained
weights, and current experiment. No asset is admitted.

### E54. 3DGrid-VQGAN: density-grid foundation pretraining with QM9 frontier readouts

**Primary sources.** The [ICLR paper](https://openreview.net/pdf/51c97777a512d94b366ffd7ba0d8979eba14e3c8.pdf),
[IBM code](https://github.com/IBM/materials/tree/main/models/3dgrid_vqgan), and
[Hugging Face checkpoint card](https://huggingface.co/ibm-research/materials.3dgrid_vqgan)
provide an unusually complete electron-density-grid foundation artifact. Its
roughly `855K` neutral PubChem pretraining corpus uses up to `30` heavy atoms,
RDKit distance-geometry/force-field conformers, five MINDO3 reoptimizations, and
the lowest-MINDO3 RHF/STO-3G density on `128^3` grids. The public code contains
pretraining, QM9 fine-tuning, inference, and embedding-extraction paths, with a
`3DGrid-VQGAN_43.pt` checkpoint name.

The QM9 downstream surface includes `gap`, `homo`, and `lumo` under
B3LYP/6-31G(2df,p), but the audited table entries `0.0088`, `0.0058`, and
`0.0057` are not explicitly unit-labelled in the available context. They are
therefore not converted to eV and are not PCQM evidence. **Grade: A/B for density
grid pretraining and artifact lineage; C for external grids, RHF/STO-3G inputs,
QM9 theory, weights, and current experiment.**

### E55. Graph2Mat: sparse equivariant density-matrix prediction

**Primary sources.** The [Graph2Mat paper](https://doi.org/10.1088/2632-2153/adc871),
[MIT-licensed repository](https://github.com/BIG-MAP/graph2mat), and [DTU data
record](https://data.dtu.dk/articles/dataset/MD17_data_for_graph2mat/26195285)
establish a public operator-learning implementation. Atomic numbers and
coordinates are mapped to sparse atom-centred density-matrix blocks with graph
construction and MACE-like equivariant embeddings. The SIESTA 5.0.0/PBE
norm-conserving pseudopotential/DZP setup covers MD17, QM9, and ethylene
carbonate examples; warm-start SCF experiments report roughly `40%` reduction.

The paper's strongest transferable idea is not a Gap number but an acceptance
protocol: electron-count error and Hamiltonian self-consistency residual are
reported as uncertainty/active-learning signals. The predicted matrix is basis
dependent and Graph2Mat exposes no direct PCQM4Mv2 B3LYP/6-31G*/ETKDG
HOMO/LUMO/Gap result. **Grade: A/B for operator targets and self-consistency
gates; C for SIESTA matrices, basis/pseudopotentials, external rows, and current
experiment.**

### E56. ACE density matrix: Grassmann-constrained electronic operators

**Primary sources.** The [ACE density-matrix paper](https://doi.org/10.1039/D5DD00230C),
[ACEsuit code](https://github.com/ACEsuit/ACEdensitymatrix), and [DaRUS data
record](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419/DARUS-4902)
provide a reproducible ACE/Julia operator route. The method expands local density
matrix blocks in an ACE basis, fits a linear model, and uses spectral retraction
onto the Grassmann manifold to preserve projector-like validity. The paper uses
18 molecules and reports one-SCF energy/force evaluation plus the commutator
residual `FD-DF` as an error and active-learning signal.

Its data contract is external: omegaB97XD/6-31G(d) and solvent/QM-MM trajectory
sources, basis-dependent 3D frames, and a large DaRUS learning-density archive.
No direct PCQM4Mv2 B3LYP/6-31G* frontier result or ETKDG path is exposed.
**Grade: A/B for Grassmann/projector constraints and operator/UQ design; C for
external density matrices, solvent/QM-MM rows, basis/theory, geometry, and current
experiment.**

### E57. SMILESDFT-CLIP/SigLIP: contrastive SMILES--density pretraining

**Primary sources.** The [NeurIPS 2025 workshop record](https://neurips.cc/virtual/2025/126037),
[IBM implementation](https://github.com/IBM/materials/tree/main/models/smilesdft_clip),
and [Apache-2.0 model card](https://huggingface.co/ibm-research/materials.smilesdft-clip)
document a paired canonical-SMILES/electron-density pretraining route. The model
uses CLIP/SigLIP-style contrastive alignment on the approximately `855K` PubChem
density pairs, reports random-SO(3)-rotation retrieval checks, and exposes
pretraining, QM9 `U0` fine-tuning, embedding extraction, and named checkpoints.

This is representation and invariance evidence rather than frontier-property
evidence. The density grids, PubChem identities, external density convention, and
absence of a matched PCQM4Mv2 B3LYP/6-31G*/ETKDG Gap result block direct import.
**Grade: A/B for multimodal pretraining and frozen-encoder controls; C for
external grids, rows, weights, downstream QM9 route, and current initialization.**

### E58. QMLearn: one-electron reduced-density-matrix gamma/delta learning

**Primary sources.** The [Nature Communications paper](https://doi.org/10.1038/s41467-023-41953-9),
[QMLearn source](https://gitlab.com/pavanello-research-group/qmlearn),
[Zenodo training data](https://doi.org/10.5281/zenodo.7946420), and [Zenodo code
snapshot](https://doi.org/10.5281/zenodo.8269767) provide a complete electronic
surrogate package. The gamma map learns a Gaussian-type-orbital 1-RDM from the
electron--nuclear external potential; the delta map predicts energies, forces,
and one-electron observables from the predicted 1-RDM or learns a residual
correction to that matrix. Fock construction and diagonalization expose
Kohn--Sham orbitals and HOMO--LUMO gaps. The paper demonstrates DFT, HF, and
full-CI surrogates on water, benzene, and alcohols, but not PCQM4Mv2.

This is a strong electronic-teacher principle rather than a transferable current
model. The data are molecule/method/basis/temperature specific, use PySCF/GTO
operator matrices and normal-mode/Eckart-frame sampling, and provide no
B3LYP/6-31G*/ETKDG PCQM route. **Grade: A/B for operator-aware teacher and
gamma-versus-delta design; C for external 1-RDM rows, code/weights, geometry, and
current experiment.**

### E59. QMLearn-SCF: optimized 1-RDM learning and force correction

**Primary sources.** The peer-reviewed [JCTC paper](https://doi.org/10.1021/acs.jctc.5c01564),
[open preprint](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/68c9a9763e708a76498380b5/original/main.pdf),
and [Zenodo record](https://zenodo.org/records/17103131) extend QMLearn with
model/hyperparameter selection, smaller training sets, and a force-correction
algorithm. KRR gamma-learning plus linear delta correction reaches roughly SCF-
threshold 1-RDM accuracy for a molecule-specific set including biphenyl. The
HOMO/LUMO/Gap curves are obtained from a Fock matrix built from the predicted
1-RDM and subsequent diagonalization along water OH-stretch and biphenyl torsion
scans; this is not direct scalar Gap regression. The package reports stabilized
AIMD/force controls and exposes reproducibility scripts/data, but the archive is
about `1.3 TB`.

**Grade: A/B for diagonalization-aware electronic-teacher acceptance, residual
correction, force/trajectory checks, and geometry-coverage design; C for external
GTO/B3LYP data, molecule-specific frames, 1-RDM labels, and current ETKDG use.**

### E60. STRUCTURES25: variational orbital-free DFT as a density/energy teacher

**Primary sources.** The [JACS paper/preprint](https://arxiv.org/html/2503.00443v2),
[official repository](https://github.com/sciai-lab/structures25), and [public
documentation](https://sciai-lab.github.io/structures25/) provide code, inference
models, data-generation tools, and a replication workflow. STRUCTURES25 represents
electron density with atom-centred basis coefficients and learns a rotationally
equivariant energy functional with tensorial Graphormer-style messages. Perturbed
effective potentials broaden the training density distribution, while variational
optimization and a fast dSAD initial guess produce stable densities. The paper
reports about `0.644 mHa` QM9 energy MAE relative to PBE/6-31G(2df,p) and
convergent density optimization, plus local-radius extrapolation on QMugs.

It predicts energies/densities rather than a direct Kohn--Sham HOMO/LUMO/Gap
head. Its explicit coordinates, density coefficients, PBE/QMugs labels, and
orbital-free software license remain external to PCQM4Mv2/ETKDG. **Grade: A/B for
variational auxiliary supervision, perturbed-state coverage, and stable density
teachers; C for external rows, weights, geometry, and current experiment.**

### E61. KineticNet: derivative-aware orbital-free electronic supervision

The [JCP paper](https://doi.org/10.1063/5.0158275) and [arXiv HTML
version](https://arxiv.org/html/2305.13316) learn kinetic-energy density and its
functional derivative from molecular quadrature-grid densities, nuclear positions,
and charges. The E(3)-equivariant point-convolution model uses atom-centred
encoding, atom--atom interaction layers, and grid decoding. BLYP/cc-pVDZ KS-DFT
with random external-potential matrix perturbations supplies off-ground-state
coverage for He, H2, H3+, HF, Ne2, and H2O; the reported test energy MAE is below
1 mHa per electron across the small systems. Density optimization is only shown
for two-electron cases, and the data are available from the authors on request.

The result is a derivative/coverage control, not a frontier-property model. **Grade:
A/B for functional-derivative supervision and off-ground-state teacher design; C
for grids, BLYP/cc-pVDZ rows, author-request data, and PCQM/ETKDG use.**

### E62. M-OFDFT: residual KEDF with gradient labels and pretrain/fine-tune controls

The peer-reviewed [Nature Computational Science paper](https://doi.org/10.1038/s43588-024-00605-8),
[arXiv PDF](https://arxiv.org/pdf/2309.16578), [Zenodo implementation](https://doi.org/10.5281/zenodo.10616893),
and [Figshare examples/checkpoints](https://doi.org/10.6084/m9.figshare.c.6877432)
provide an unusually complete orbital-free reference. M-OFDFT uses atom-centred
even-tempered density coefficients, Graphormer attention, local frames, overlap-
matrix reparameterization, and an atomic-reference gradient offset. It trains on
multiple density-coefficient states from KS SCF iterations and their projected
gradients, not only one ground-state energy per molecule. The in-scale model
learns a residual on an APBE KEDF and reports PBE/6-31G(2df,p) energy MAE of
`0.18 kcal/mol` on ethanol and `0.93 kcal/mol` on QM9. On a chignolin
pretraining/fine-tuning test, accessible-scale pretraining reduces the error by
`35.4%` relative to scratch training, while the final workflow also uses 500
large-system structures for fine-tuning. The reported empirical scaling is
`O(N^1.46)` versus `O(N^2.49)` for KSDFT, with up to `27.4x` speedup on a
738-atom protein.

This is the strongest new evidence for a residual target plus gradient-landscape
teacher and an explicit pretrain/from-scratch/fine-tune control. It still has no
direct HOMO/LUMO/Gap head; all density coefficients, coordinates, PBE/QM9/QMugs/
MD17 labels, and external weights remain outside PCQM4Mv2/ETKDG. **Grade: A/B for
residual/gradient/pretraining protocol design; C for external electronic rows,
explicit 3D geometry, and current experiment.**

### E63. Meyer--Weichselbaum--Hauser: derivative supervision with an explicit failure mode

The open [JCTC article](https://doi.org/10.1021/acs.jctc.0c00580) and [PMC full
text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7482319/) study a 1D
non-interacting-fermion model with 100 analytic training potentials and 1,000
test potentials. KRR, CNN, and ResNet models trained jointly on kinetic energy and
its functional derivative improve derivative error and iterative-density
stability, but unconstrained optimization still leaves the valid region because
of noisy ML derivatives; PCA projection or a von-Weizsaecker penalty is needed.
This is retained as a negative control for auxiliary-gradient proposals, not as
a molecular or PCQM experiment. **Grade: C for direct admission; A/B only for
the failure-aware derivative-loss and on-manifold acceptance lesson.**

### E64. Hamiltonian, self-consistency, and operator-level Delta continuation

The companion [Hamiltonian/self-consistency deep-reading record](deep_reading_hamiltonian_self_consistency_2026-09-07.md)
audits the next method family without changing the scalar-label contract. The
main evidence is now organized as follows:

- [QH9](https://arxiv.org/html/2306.09549), [AIRS/QHBench](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9),
  [nablaDFT](https://github.com/AIRI-Institute/nablaDFT), and [∇²DFT](https://arxiv.org/html/2406.14347)
  establish public Hamiltonian/density data and loaders. They are useful for
  schema, operator acceptance, and teacher-quality checks, but their
  B3LYP/def2-SVP or ωB97X-D/def2-SVP labels and explicit 3D geometries are not
  the current B3LYP/6-31G*/ETKDG labels.
- [Self-Consistency Training](https://proceedings.mlr.press/v235/zhang24ak.html),
  [DEQHNet](https://github.com/Zun-Wang/DEQHNet), and
  [NeuralSCF](https://github.com/songfeitong/neuralscf) turn the density/Hamiltonian
  residual into a differentiable fixed-point or SCF objective. The transferable
  lesson is to measure self-consistency and post-diagonalization frontier error,
  not to import their basis-dependent matrices.
- [HamEvo](https://arxiv.org/html/2606.14498), its [official code](https://github.com/axdfhj/HamEvo_official),
  and [dataset surface](https://huggingface.co/datasets/ZJUSCL/hamevo-data) provide
  the clearest recent operator-level Delta design: a learned local `Delta H`,
  fixed-point iteration, Anderson/Broyden differentiation, and staged SCF
  transition pretraining. This is a method reference only; it is not a scalar
  `Delta Gap` result and its external basis/functionals/coordinates remain out
  of scope.
- [QHFlow2](https://arxiv.org/html/2602.16897), [official code](https://github.com/seongsukim-ml/QHFlow2),
  and [QH9 checkpoints](https://huggingface.co/ksusu/QHFlow2-QH9) are the most
  useful public frontier-aware operator checkpoint in this batch: HOMO/LUMO/Gap
  are obtained after predicting H and diagonalizing it. The checkpoint is still
  a QH9 B3LYP/def2-SVP explicit-3D artifact, not a legal current initialization.
- [HELM](https://arxiv.org/html/2510.00224) gives the cleanest Hamiltonian-
  pretraining versus frozen/fine-tuned head comparison, but its paper marks
  code/data as forthcoming. [QHNetV2](https://arxiv.org/html/2506.09398),
  [SPHNet](https://proceedings.mlr.press/v267/luo25l.html), and
  [WANet/WALoss](https://arxiv.org/pdf/2502.19227) add efficiency, sparse-gate,
  overlap-aware loss, and matrix-quality controls, without a same-contract
  PCQM/ETKDG frontier result.

**Admission decision.** These sources are retained as A/B evidence and as
future teacher/pretraining/delta protocol references. No QH9/∇²DFT row,
Hamiltonian checkpoint, external coordinate, or remote experiment is admitted.
Any later scalar-Delta experiment must use the unchanged PCQM or repaired-2M
database, an immutable ETKDG cache, a same-contract random-init control, and a
separate acceptance record; `Delta H` must not be relabelled as `Delta Gap`.

### E65. Differentiable Hamiltonians, SAP residuals, density teachers, and completed operator artifacts

Five further sources close several method-definition gaps:

- [Suman et al.](https://doi.org/10.1021/acs.jctc.5c00522) plus
  [PySCFAD](https://github.com/fishjojo/pyscfad) makes the Hamiltonian an
  intermediate layer and differentiates orbital/response losses. It supports
  multi-target and basis-transfer design, but its QM7/QM9 and STO-3G/def2-TZVP
  contract is external.
- [SAP Hamiltonian learning](https://arxiv.org/html/2606.12326) is the clearest
  operator-level residual definition in this extension: `Delta F` is the
  converged PBE0 Fock matrix minus a screened SAP baseline. Its QM9 curves are
  evidence for operator Delta-learning, not a scalar PCQM Gap result.
- [Direct density-matrix prediction](https://doi.org/10.1021/acs.jctc.4c00042)
  provides a small-system SCF-initialization/energy/force control. It is useful
  for acceptance criteria and a negative test for matrix-MAE-only claims, but
  not for importing a BLYP/cc-pVDZ matrix teacher.
- [HamGNN](https://doi.org/10.1038/s41524-023-01130-4) with
  [code](https://github.com/QuantumLab-ZY/HamGNN) and
  [Zenodo data/models](https://doi.org/10.5281/zenodo.8157128) is a completed
  E(3)-equivariant operator artifact. [HAMSTER](https://doi.org/10.1038/s41467-026-70865-7)
  with [MIT code](https://github.com/TheoFEM-TUM/Hamster.jl) and
  [Zenodo data](https://doi.org/10.5281/zenodo.18485403) is a completed physical
  model-to-DFT Hamiltonian Delta workflow. Both are operator/materials assets,
  not current scalar PCQM models.

**Admission decision.** These five sources are recorded as A/B method or
artifact evidence. They do not alter the database, geometry source, active
random-init screen, or remote authorization. Only a future protocol that
defines a same-level, same-identity baseline and distinguishes `Delta Gap`,
`Delta F`, and `Delta H` can be considered.

### E66. Same-database electronic and tetrahedral pretraining controls

Two additional pretraining papers were added because they use the existing
PCQM4Mv2 molecule pool while keeping their method roles explicit:

- [Graphormer atom-in-a-molecule quantum pretraining](https://doi.org/10.1186/s13321-025-00970-0)
  compares atom-level charge/NMR/Fukui supervision, PCQM molecular HLG
  pretraining, masking, and scratch. Its strongest transfer evidence is for
  local atom-level quantum supervision rather than HLG-only initialization;
  the public [GraphQPT code](https://github.com/aidd-msca/GraphQPT) exposes
  configs and representation analyses. It is an ADMET/HLM transfer study, not
  a direct PCQM Gap improvement.
- [Tetrahedral Molecular Pretraining](https://doi.org/10.1016/j.patcog.2025.112638)
  uses PCQM 3D structures for label-free tetrahedral orientation/reconstruction
  pretraining and reports `0.0817` versus `0.0910` in its PCQM downstream
  comparison. The [MIT implementation](https://github.com/sunyuancheng/Tetrahedral-Molecular-Pretraining)
  and checkpoint make it a completed artifact, but DFT-coordinate input,
  older dependencies, and unresolved downstream validation lineage prevent
  importing the result into the ETKDG contract.

**Admission decision.** Both are reliable post-selection pretraining/teacher
references. Neither changes the database, the current random-init architecture
screen, or remote authorization. A future same-database probe must preserve a
fresh random-init control and must report the teacher geometry, pretraining
identity role, and inference-time ETKDG contract separately.

### E67. Implementation-backed denoising continuation: SCD and DenoiseVAE

Two pretraining leads deserve separate treatment because their evidence is
strong enough for a reproducibility audit, but not for direct import:

- [Self-Conditioned Denoising](https://arxiv.org/html/2603.17196v1) publishes
  the clean-embedding-conditioned denoising objective, an [official
  implementation](https://github.com/TyJPerez/SelfConditionedDenoisingAtoms),
  and the `ct-scd-pcq` [PCQM checkpoint](https://huggingface.co/Ty-Perez/ct-scd-pcq).
  The reported downstream frontier results are on QM9 after PCQ pretraining,
  so the evidence supports a representation lead, not a direct PCQM Gap gain.
- [DenoiseVAE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html)
  reports a direct PCQM4Mv2 validation value of `0.0777 +/- 0.0005` with
  `1.44M` parameters and releases [code](https://github.com/liuyurou1/DenoiseVAE).
  The public code audit found a GEOM default, an RDKit Embed+MMFF PCQM builder,
  no visible checkpoint/license manifest, and an undefined return variable;
  therefore it is a paper/method audit, not a runnable asset.

**Admission decision.** SCD is the first evidence-only hash and forward-pass
smoke candidate; DenoiseVAE is a direct-PCQM paper audit with a code-quality
negative control. Neither authorizes checkpoint reuse, a remote job, or a
database/geometry change. Any future adaptation must regenerate both clean and
corrupted inputs with ETKDG and include a fresh random-init control.

### E68. Interaction teachers: IEM and MolInteract

[IEM](https://www.ijcai.org/proceedings/2024/675) is a same-PCQM image-teacher
route. It pretrains 2D/3D image encoders on about 2M molecules with image
contrastive learning and atom, bond, geometry, and property distribution tasks,
then distills frozen teacher outputs into a graph student. Its graph-only
inference path is the useful part for MolGap. The [official code](https://github.com/HongxinXiang/IEM)
exposes preprocessing, distillation, and teacher-asset paths, but the audited
3D pipeline is RDKit/MMFF-oriented and the checkpoint/data/license manifest is
not closed. Its reported downstream gains are MoleculeNet results rather than a
direct PCQM Gap result.

[MolInteract](https://kdd2025.kdd.org/wp-content/uploads/2025/07/paper_22.pdf)
keeps the 2D GINE and 3D SchNet streams in repeated interaction layers rather
than fusing only final predictions. Its PCQM pretraining uses distance, angle,
dihedral, bond, shortest-path, and centrality relation tasks, while the
reported frontier table is on QM9. No author-maintained code or checkpoint was
located in the audit, and the original 3D-dependent model is not shown to use
ETKDG consistently at train and inference.

**Admission decision.** IEM is a B-level teacher/distillation reference and
MolInteract is a B-level mechanism/C-level artifact reference. Neither
reopens late prediction fusion, authorizes external images or conformers, or
changes the current database. A future test would need ETKDG-only views,
graph-only student inference, a frozen-teacher manifest, and an identical
random-init control.

### E69. LeJEPA: predictor-free pretraining as a control lesson

[LeJEPA](https://arxiv.org/abs/2609.04261) combines a predictor-free
joint-embedding predictive architecture with SIGReg and evaluates GPS and
Chemprop encoders. It is not a PCQM or quantum-property paper: its tasks are
ogbg-molhiv and antibiotic activity. Its useful result is methodological: the
paper separates frozen-probe performance from end-to-end fine-tuning and shows
that a strong representation probe need not imply a robust fine-tuning gain.
The paper claims code/config/checkpoint release, but no independently
retrievable repository or checkpoint manifest was closed in this audit.

**Admission decision.** Retain LeJEPA as a recent algorithm and negative-control
reference. Any future MolGap pretraining report must include scratch,
frozen-probe, fine-tuned, seed, and representation-combination controls. No
LeJEPA artifact, data, or initialization enters MolGap.

### E70. Pretraining negative control: Does GNN Pretraining Help Molecular Representation?

The [NeurIPS 2022 paper](https://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html)
is a useful control because it varies pretraining objectives, data splits,
input features, pretraining scale, and GNN architecture rather than reporting a
single favorable recipe. Its official abstract reports that self-supervised
pretraining is often not statistically better than scratch, supervised gains
can shrink with richer features or balanced splits, and hyperparameters can
matter more than the objective on small molecular tasks. The paper uses
ZINC15/SAVI/ChEMBL-style sources and MoleculeNet downstream tasks, not
PCQM4Mv2 HOMO/LUMO/Gap.

**Admission decision.** Keep this as a mechanism/negative-control source. It
does not support a MolGap gain, but it makes scratch, frozen-probe,
fine-tuning, feature, and optimizer controls mandatory for any later
same-database pretraining test. No code, checkpoint, data, or experiment is
added.

### E71. ET-OREO: force-centric equilibrium/off-equilibrium pretraining

The [NeurIPS 2023 ET-OREO paper](https://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html)
unifies equilibrium and off-equilibrium 3D pretraining. It uses zero-force
regularization and force-guided denoising for equilibrium structures, direct
force learning for off-equilibrium structures, and reports more than 15M
conformations assembled from PCQM4Mv2, ANI1x, MD17, and Poly24. The paper
reports approximately three-fold force-accuracy improvement over an
unpretrained Equivariant Transformer and improved/fast molecular-dynamics
behavior, while property transfer is described as competitive. These are
force/MD and general property-transfer results, not a direct PCQM Gap score.

The transferable idea is the clean-geometry versus perturbed-geometry
consistency objective. The incompatibilities are equally important: force
labels and explicit non-ETKDG conformers are outside the current contract, and
the model is an expensive equivariant 3D route. A public author-maintained
code/checkpoint was not independently located in this audit.

**Admission decision.** Retain ET-OREO as a B-level geometry/pretraining
reference. A future ETKDG-only corruption objective could be specified after
architecture selection, but this paper does not authorize force labels,
external conformers, or a new database.

### E72. JMP: supervised multi-domain pretraining at scale

The [ICLR 2024 JMP paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a896a8bb065774169d9ad65f19208b7-Abstract-Conference.html)
trains a shared model as named supervised tasks over about 120M systems from
OC20, OC22, ANI-1x, and Transition-1x. It reports a 59% average improvement
over scratch and SOTA or matched results on 34/40 downstream tasks, including
QM9, rMD17, MatBench, QMOF, SPICE, and MD22. The [official repository](https://github.com/facebookresearch/JMP)
exposes configs, preprocessing, fine-tuning commands, and JMP-S/JMP-L
checkpoint links; it also says data are not bundled, the repository is
deprecated/archived, and most of the code is CC-BY-NC.

JMP is strong engineering evidence, but it is not a matched PCQM4Mv2 Gap
result. The data, force/energy labels, explicit 3D geometries, and license
terms differ from MolGap. **Admission decision:** retain the per-corpus task
map, normalization, sampling, scratch-control, and cost-accounting practices;
do not import the corpus or checkpoints and do not change the current database.

### E73. CSI: task-aligned pretraining data versus scale

The TMLR 2026 [CSI paper](https://arxiv.org/abs/2502.11085), *On the Importance
of Pretraining Data Alignment for Atomic Property Prediction*, introduces a
Chemical Similarity Index inspired by FID. It reports that a carefully selected
task-aligned dataset can match or exceed large mixed pretraining at one
twenty-fourth of the budget, and that adding poorly aligned data can hurt.
The [official `efficient-atom` repository](https://github.com/Yasir-Ghunaim/efficient-atom)
contains feature extraction, upstream/downstream structure mapping, CSI
calculation, individual-versus-mixed pretraining, class-balanced sampling,
fine-tuning, and checkpoint handoff instructions.

The source set is the JMP atomic-property suite (OC20/OC22/ANI1x/Transition1x
upstream and rMD17/QM9/MD22/QMOF/SPICE/MatBench downstream), not PCQM4Mv2 Gap.
Its useful MolGap hypothesis is therefore narrow: a train-role-only
PCQM4Mv2 or repaired-2M selector could be compared with an equal-budget random
subset and a full same-database pretraining control. The selector must be
frozen without official validation/test-dev molecules or target labels.

**Admission decision.** Retain CSI as an A/B data-selection and budget-control
reference. The same-database adaptation is hypothetical and requires a new
protocol; no external data, code dependency, checkpoint, or experiment enters
the current route.

### E74. Synchronizing the earlier geometry-pretraining and delta reserve

The unique ledger now also indexes sources that were already deep-read in the
geometry, public-code, denoising, and teacher/delta records but had not yet
received a row in the single-number authority. The source-level decisions are
unchanged:

| Newly synchronized source | Audited evidence | Bounded decision |
|---|---|---|
| [GraphMVP](https://arxiv.org/html/2110.07728) | Contrastive plus variational 2D/3D representation alignment, with 3D discarded for downstream inference and public code | ETKDG-only teacher/student template; no current initialization |
| [GeoSSL-DDM](https://arxiv.org/html/2206.13602) | SE(3)-invariant pair-distance denoising and multi-task geometry transfer | Distance-denoising objective reference; no all-pair geometry in the active screen |
| [UnifiedMolPretrain](https://arxiv.org/html/2207.08806) | About 3.38M PCQM4Mv2 pretraining examples, masked 2D/3D reconstruction, and public code | Same-database multimodal reference; no direct PCQM Gap result or DFT-coordinate import |
| [VideoMol](https://www.nature.com/articles/s41467-024-53742-z) | About 2M PCQM molecules rendered as 60-frame videos with public code/data/model surfaces | Multi-view artifact-manifest reference; no raster/video input route |
| [PVD](https://arxiv.org/html/2206.00133) | Direct PCQM denoising configuration, checkpoint, and separated random-init/Noisy-Nodes/pretraining controls | Future ETKDG-only denoising protocol; no checkpoint warm start |
| [D&D](https://ojs.aaai.org/index.php/AAAI/article/view/31986) | Frozen 3D teacher with graph- and node-level distillation on PCQM geometry | Strong teacher contract; DFT conformer and absent artifact block current use |
| [DelFTa](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d2cp00834c) | Public GFN2-xTB-to-DFT residual workflow with direct-versus-delta and cost/baseline-correlation evidence | Same-database proxy delta reference; no external target or geometry import |

These records complete the indexing gap without changing the database, the
ETKDG rule, or the random-init architecture boundary. The full readings remain
in [the geometry/pretraining record](deep_reading_2026-09-07.md), [the denoising
continuation](deep_reading_denoising_continuation_2026-09-07.md), [the
teacher/delta continuation](deep_reading_teacher_delta_continuation_2026-09-07.md),
and [the public-code audit](deep_reading_public_code_2026-09-07.md).

## Decision matrix for the remaining search

| Route | New information | Overlap with closed routes | Contract class | Priority |
|---|---|---|---|---:|
| HamEvo operator Delta | Local SCF-like `Delta H`, fixed-point refinement, Anderson/Broyden implicit differentiation, and staged transition pretraining | Different target object from scalar Gap; external basis/functional/3D roles | Post-selection operator-teacher or Hamiltonian-residual method; never a scalar Delta claim without a separate baseline | P1 method audit |
| QHFlow2 frontier Hamiltonian checkpoint | Public H/S checkpoint with HOMO/LUMO/Gap readout, stable/dynamic QH9 splits, and direct H-to-energy/force path | Explicit 3D and external QH9 theory; not the current ETKDG scalar route | Frozen teacher/probe only after identity, theory, and geometry audit | P1 asset audit |
| Self-Consistency Training / DEQHNet / NeuralSCF | Differentiable eigen/density residuals, fixed-point layers, and SCF trajectory pretraining | Matrix/density labels and basis-specific geometry are absent from current PCQM | Self-consistency acceptance and auxiliary-teacher design; no current matrix import | P1 method reference |
| QH9 / nablaDFT / ∇²DFT assets | Public Hamiltonian/density schemas, loaders, large conformer databases, and operator metrics | B3LYP/def2-SVP or ωB97X-D/def2-SVP, explicit 3D, and external identity surfaces | Schema/acceptance/OOD reference; preserve current database | P2 asset audit |
| HELM Hamiltonian pretraining | Controlled pretrain/frozen-head/fine-tune comparison and low-data energy evidence | Code/data TBA; OMol_CSH/∇²DFT energy contract and no direct frontier result | Pretraining protocol reference only; no checkpoint or rows | P2 |
| QHNetV2 / SPHNet / WANet | SO(2) local frames, sparse adaptive gates, overlap-aware loss, and efficiency/quality diagnostics | Operator matrices, basis dependence, and no current ETKDG scalar Gap result | Efficiency and loss-design reference; no current experiment | P2 |
| PySCFAD / differentiable Hamiltonian | Effective H as a differentiable intermediate with orbital/response multi-target losses and reduced-basis transfer | QM7/QM9, STO-3G/def2-TZVP, explicit integral/eigensolver layer | Differentiable ML/QM protocol reference; no current scalar import | P2 |
| SAP `Delta F` | Screened SAP baseline, symmetry-adapted orbital basis, core/valence residual GNN, and frontier-aware QM9 curves | PBE0/cc-pVDZ/cc-pVTZ operator matrices and explicit basis; no public checkpoint | Operator Delta reference; no scalar PCQM Delta claim | P1 method reference |
| Direct density-matrix teacher | Public paper evidence for SCF initialization, non-self-consistent energy/force, and matrix-MAE/physical-stability tradeoff | BLYP/cc-pVDZ, molecule-specific small systems, dense matrices | Teacher acceptance/negative control only | P2 |
| HamGNN / HAMSTER | Completed public Hamiltonian code/data/model artifacts; HAMSTER supplies a physics-model-to-DFT residual path | Tight-binding/materials/periodic PBE/SOC/VASP roles, explicit 3D and non-PCQM labels | Artifact/Delta packaging reference; no current import | P2 |
| GraphQPT atom-level quantum pretraining | Charges/NMR/Fukui versus PCQM HLG versus masking controls, public Graphormer code, and representation/spectral diagnostics | External atom-level QM geometry/theory, ADMET/HLM downstream, and no direct PCQM Gap gain | Local-electronic teacher/pretraining attribution reference; no current initialization | P1 post-selection |
| Tetrahedral Molecular Pretraining | Same-database PCQM 3D tetrahedral orientation/reconstruction pretraining, public code/checkpoint, and a reported downstream Gap comparison | DFT coordinates, large Transformer-M scale, old stack, and unresolved split/validation lineage; not ETKDG-equivalent | Frozen teacher or ETKDG-reproduced pretraining reference; no current initialization | P1 post-selection |
| SCD | Clean-embedding-conditioned denoising, official implementation, and public PCQM checkpoint | Equilibrium-coordinate TorchMD-Net route, QM9 downstream rather than direct PCQM Gap, and no ETKDG equivalence | Evidence-only checkpoint/hash/smoke audit; later ETKDG-only objective adaptation | P1 audit |
| DenoiseVAE | Direct PCQM validation claim, adaptive atom-wise noise, and public paper/code | RDKit+MMFF builder, GEOM default, missing checkpoint/license, broken return variable, and no ETKDG route | Paper/method and code-negative-control audit; no current initialization | P1 audit |
| IEM | Same-PCQM image teacher, five distribution/contrastive tasks, frozen-teacher graph distillation, and graph-only student inference | RDKit/MMFF-oriented rendering, incomplete checkpoint/data/license closure, MoleculeNet rather than direct PCQM Gap | ETKDG-only teacher/student protocol after artifact and identity audit; no images or weights now | P1 post-selection |
| MolInteract | Repeated 2D GINE--3D SchNet interaction with reciprocal geometry/topology tasks and PCQM pretraining | QM9 frontier downstream, 3D-dependent original model, no verified code/checkpoint, no ETKDG equivalence | Cross-modal interaction reference only; separate 2D student and cost control required | P2 method audit |
| LeJEPA | Predictor-free joint embedding plus SIGReg and frozen-probe/fine-tune comparison | ogbg-molhiv/antibiotic tasks, no PCQM Gap evidence, no independently closed artifact | Pretraining evaluation-control reference; no current initialization | P2 observation |
| Does GNN Pretraining Help Molecular Representation? | Controlled objective/split/feature/scale/architecture ablations and a negative self-supervised-pretraining lesson | ZINC15/SAVI/ChEMBL and MoleculeNet rather than PCQM Gap; no verified code/checkpoint | Mandatory scratch/frozen-probe/fine-tune and feature-control reference; no current initialization | P1 control |
| ET-OREO | Force-centric equilibrium/off-equilibrium denoising and direct force supervision over >15M conformations | External force labels, ANI1x/MD17/Poly24, explicit 3D geometry, no direct PCQM Gap, and no verified artifact | ETKDG-only denoising objective reference after architecture selection; no force/data import | P1 method audit |
| JMP | Public large-scale supervised multi-domain pretraining, per-dataset tasks, configs, and named checkpoints | OC20/OC22/ANI1x/Transition1x, external 3D theories, CC-BY-NC/archived code, and no direct PCQM Gap | Pretraining bookkeeping and external teacher reference; no corpus/weights/current initialization | P2 |
| CSI / efficient-atom | Chemical Similarity Index, feature-based upstream selection, equal-budget comparison, and public scripts | JMP atomic-property data, no same-database PCQM Gap result, and no ETKDG contract | Hypothetical train-role-only same-database data-alignment protocol; no current run | P1 method audit |
| GraphMVP | Contrastive plus variational 2D/3D alignment with graph-only downstream inference | Privileged paired conformers, old stack, no direct PCQM Gap result | ETKDG-only teacher/student representation-alignment protocol; no current initialization | P2 |
| GeoSSL-DDM | Pair-distance score matching as an SE(3)-invariant geometry-denoising proxy | All-pair distances, QM9/MD17/LBA/LEP, non-ETKDG conformer assumption | Distance-denoising objective reference only | P2 |
| UnifiedMolPretrain | Same-PCQM masked 2D/3D reconstruction and cross-modal generation with public code | DFT/paired 3D roles, downstream MoleculeNet/OGB tasks, no direct PCQM Gap score | Multimodal pretraining reference; no geometry or weight import | P2 |
| VideoMol | PCQM conformer-to-video preprocessing, public processed data/code/models, and multi-view aggregation | RDKit/MMFF/PyMOL rendering, view dependence, storage/CPU cost, no direct Gap result | Manifest and multi-view artifact reference only | P3 |
| D&D | Frozen 3D denoising teacher with global or atom-level student alignment | Lowest-energy DFT conformers, no verified code/checkpoint, non-ETKDG teacher | ETKDG-only distillation contract after a new protocol | P1 teacher reference |
| DelFTa | Public low-fidelity GFN2-xTB residual workflow and direct/delta cost comparison | QMugs, ωB97X-D/def2-SVP, explicit 3D and required reference calculation | Same-database proxy/residual method reference; no current run | P1 delta audit |
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
| DGT | Public dual atom/bond Transformer; PCQM4Mv2 Gap pretraining at 10K/100K/1M subsets; MIT code, Zenodo package, and Figshare source data; QM9 downstream HOMO/LUMO/Gap transfer | Main reported downstream evidence is QM9, optional 3D uses DFT/MMFF/UFF-derived geometry, dense atom/bond pair matrices are quadratic, and no direct official PCQM Gap score is exposed | Complete pretraining/architecture reference; adapt only under an ETKDG-only role map and a no-pretraining control | P1 audit/reference |
| QuantumCanvas | Public two-body electronic/charge/orbital fields, eight-model benchmark, composition-held-out split, and QM9/MD17/CrysMTM transfer | SCC-DFTB/DFTB+ PTBP labels, diatomic explicit coordinates, paper/README schema discrepancy, no direct PCQM result | Electronic auxiliary-teacher and artifact-manifest reference; no rows, image tensors, or weights | P2 audit/reference |
| DTU solar / tmQM / BOS-TMC | Completed public B3LYP or DFT frontier-orbital databases across organic donor--acceptor and transition-metal domains | Basis, geometry, charge/spin, element, experimental-coordinate, and license roles differ from PCQM/ETKDG; overlap must be measured | External OOD/theory-shift/teacher audit only; keep database unchanged | P2/P3 |
| TMC-Delta-ML | Public benchmark-versus-Delta switch, low-fidelity graph archive, CSD-ID splits, and HOMO--LUMO target path | Transition-metal tmQMg, GFN2-xTB geometry, u-NatQG, PBE0-D3BJ/def2-TZVP or LSDA/LANL2DZ, no current PCQM result | Δ-learning protocol and cost-control reference; no current rows, features, or run | P1 method reference |
| SelectedML | Public structure-based chemical classes, direct/Delta KRR scripts, three-seed learning curves, and QM7/QM9 HOMO/LUMO/Gap evidence | QM7/QM9, KRR CM/BoB/SLATM, GW/B3LYP/ZINDO roles, no PCQM/ETKDG result | Family-conditioned residual/OOD diagnostic; possible post-selection mixture-of-experts reference, no current model split | P2/P3 |
| HLP-Stack | Public 2D/3D descriptor stacking, raw/processed data, notebooks, saved models, and near-perfect QM9 HOMO/LUMO claims | QM9 B3LYP/6-31G(2df,p), DFT-derived 3D descriptors, target-adjacent feature path, no PCQM/ETKDG result | Negative descriptor/leakage audit only; no model/feature import | Exclude |
| PET-MAD-DOS | Public DOS/band-gap model, fixed energy grid, electron count/mask schema, reproduction data/scripts, UQ/LoRA, and checkpoint | MAD/external materials and molecule distribution, PBEsol/Quantum Espresso, DOS-derived gap, non-ETKDG geometry | Electronic-teacher packaging and structured DOS/UQ reference; no rows/weights/current experiment | P2 audit/reference |
| FieldMACE | Public long-range multipole-message code/checkpoints/data, foundation transfer, and explicit multipole-order controls | QM/MM point charges, solvated/excited-state energy/force targets, 5 Å local graph, no HOMO/LUMO/Gap/ETKDG result | Long-range message and transfer design reference; no MM/multipole features or current run | P2 |
| MACE-POLAR-1 | Public charge/spin-aware foundation checkpoints; 100M OMol25 training; multipolar field and Fukui equilibration | `omegaB97M-V` OMol25 3D energy/force model, no PCQM frontier head/result, external ASL weights and rows | Physically constrained electronic-teacher/foundation reference; no current initialization or data import | P2 |
| MACE-H | Public operator-learning code/configs/containers; high-body-order KS Hamiltonian and band/DOS post-processing | Periodic 2D/bulk-Au OpenMX/FHI-aims basis/SOC contract, no molecular PCQM/ETKDG Gap result | Operator-level/high-body-order mechanism reference; no current molecular experiment | P2/P3 |
| CheMeleon | Public descriptor-pretrained D-MPNN, one-million-molecule training surface, released weights, and Chemprop transfer path | PubChem external corpus, Mordred descriptor targets, graph/SMILES input, no matched PCQM B3LYP/6-31G* Gap result, and high foundation-pretraining cost | Same-database train-only descriptor-pretraining control reference; no external rows/weights/current initialization | P2 audit/reference |
| Zatom-1 | Public 3D flow foundation code/checkpoints, QM9 `homo`/`lumo`/`gap` property route, frozen-trunk/LoRA/non-pretrained controls | QM9/Matbench/OMol25/MPtrj external roles, explicit 3D/lattice inputs, no matched PCQM B3LYP/6-31G* Gap/ETKDG result, large cost | 3D teacher/foundation and transfer-control reference; no current weights, rows, or initialization | P2 |
| OrbNet-Equi | Public AO-electronic feature network, GFN-xTB residual route, QM9 HOMO/LUMO/Gap direct-versus-Delta curves, and Zenodo source/code/data | GFN-xTB operators, QM9/SDC21 external theory/geometry, no matched PCQM B3LYP/6-31G*/ETKDG result, high feature cost | Electronic-feature and Delta-control reference; no current features, rows, weights, or initialization | P1 method reference |
| Image-super-resolution density | Public real-space density model/data/weights; QM9 one-step HOMO/LUMO/Gap and conformer/element transfer controls | PBE/GTH-TZV2P density labels, uniform grids, one-step diagonalization, QM9/water/MD roles, no direct PCQM/ETKDG head | Density-teacher and post-diagonalization error reference; no current grid data or model | P1 post-selection |
| 3DGrid-VQGAN | Public `855K` density-grid pretraining, VQGAN checkpoint, IBM fine-tuning/inference code, and QM9 HOMO/LUMO/Gap readouts | RHF/STO-3G density grids, MINDO3/RDKit source geometries, QM9 B3LYP/6-31G(2df,p), unit-ambiguous table values, no PCQM/ETKDG result | Density-grid foundation and frozen-teacher reference; no grids, weights, or eV headline import | P1 post-selection |
| Graph2Mat | Public sparse density-matrix predictor, SIESTA/QM9/MD17/EC data, SCF warm-start, electron-count and self-consistency diagnostics | SIESTA basis/pseudopotentials, matrix blocks, external coordinates, no direct frontier head or PCQM/ETKDG result | Operator-teacher quality gate and self-consistency/UQ reference; no matrix rows or current experiment | P1 post-selection |
| ACE density matrix | Public ACE/Grassmann density-matrix model, Julia code, DaRUS learning/prediction archives, commutator residual | omegaB97XD/6-31G(d), solvent/QM-MM trajectories, basis-dependent 3D density blocks, no direct PCQM Gap result | Projector-constrained operator/UQ reference; no external density frames or weights | P1 post-selection |
| SMILESDFT-CLIP/SigLIP | Public paired SMILES--density contrastive pretraining, rotation retrieval checks, IBM code and checkpoints | PubChem density grids, external density convention, QM9 `U0` downstream, no PCQM B3LYP/6-31G*/ETKDG frontier result | Frozen multimodal encoder and invariance-control reference; no external pairs/weights | P2 |
| QMLearn | Public gamma/δ 1-RDM surrogate, QMLearn source, Zenodo data/code, and post-Fock HOMO/LUMO/Gap readouts | GTO/PySCF, molecule/method/basis/temperature-specific 1-RDM labels, normal-mode/Eckart geometries, no PCQM/ETKDG result | Operator-aware electronic teacher and gamma-versus-delta control; no external 1-RDM rows or weights | P1 post-selection |
| QMLearn-SCF | Peer-reviewed optimized 1-RDM workflow, KRR/linear-δ controls, force correction, torsion/geometry scans, and 1.3-TB archive | Molecule-specific GTO/B3LYP data, post-diagonalization frontier levels, external AIMD/force roles, no direct PCQM Gap head | Teacher-quality, residual, force/coverage acceptance reference; no current data or initialization | P1 post-selection |
| STRUCTURES25 | Public variational orbital-free DFT, perturbed-potential data generation, density optimization, QM9/QMugs energy/density evidence, and inference models | PBE/6-31G(2df,p), QMugs, atom-centred density coefficients, explicit 3D and no direct frontier head | Variational density/energy auxiliary teacher and perturbation-coverage reference; no external density coefficients or weights | P1 post-selection |
| KineticNet | E(3)-equivariant grid-to-grid kinetic-energy and derivative fields, perturbed external potentials, and small-system chemical-accuracy evidence | BLYP/cc-pVDZ, real-space grids, two-electron-only density optimization, no direct frontier head or public archive | Derivative supervision and off-ground-state coverage reference; no current grid/teacher experiment | P2 |
| M-OFDFT | Public orbital-free implementation with APBE residual, projected-SCF gradient labels, density optimization, pretrain/from-scratch/fine-tune comparison, and protein extrapolation | PBE/6-31G(2df,p), QM9/QMugs/MD17/chignolin density coefficients and explicit 3D geometry; no direct PCQM Gap head | Strong residual/gradient/pretraining protocol reference; no external rows, coefficients, weights or current experiment | P1 method reference |
| Meyer derivative training | 1D KRR/CNN/ResNet joint energy-plus-functional-derivative training and explicit noisy-derivative failure analysis | Synthetic 1D fermion system, no molecule, geometry, frontier target, or PCQM result | Negative control for derivative-loss and on-manifold acceptance; no experiment | P3 |
| PVD | Direct PCQM4Mv2 self-supervised denoising, official checkpoint/config, QM9 HOMO/LUMO/Gap transfer, and random-init/Noisy-Nodes/pretraining controls | DFT-equilibrium coordinates, full-PCQM upstream role, TorchMD/GNS stack, and no same-contract PCQM Gap result | Completed denoising reference; future ETKDG-only protocol after architecture selection | P1 audit/reference |
| Frad / FradNMI | Chemical-aware RN/VRN noise, fractional CGN target, public code/weights/data, QM9 frontier and force/robustness ablations | DFT main coordinates, RDKit+MMFF robustness geometry, legacy TorchMD stack, QM9 downstream, and closed torsion-state route | Chemical-aware pretraining reference; future ETKDG-only audit only | P1 audit/reference |
| SliDe | BAT bond/angle/torsion pretraining, random-sliced force objective, public code/models, PCQM force audit and QM9 frontier ablations | DFT equilibrium geometry, OpenFF/Sage prior, QM9 rather than PCQM Gap downstream, GET/Nv cost | Future ETKDG-only physics-informed pretraining reference | P1 audit/reference |
| CCMD | Direct PCQM validation of global/local 3D-to-2D distillation and size-normalized atom loss | DFT teacher coordinates, prior split, large Graphormer, no verified code/checkpoint | Future ETKDG teacher-loss control design; no current score | P1 teacher reference |
| NVIDIA PCQM4Mv2 winner | Completed direct PCQM challenge/validation evidence, heterogeneous modalities, OOF Huber stacking, and size-shift diagnostic | Provided PCQM 3D SDF/RDKit images, train+valid fold construction, 39-checkpoint/10-model compute, no ETKDG single-model comparability | Delivery-time ensemble/OOF protocol reference; no current score or artifact import | P1 delivery reference |
| OSCAgent | Graph/SMILES symmetric InfoNCE plus auxiliary LUMO regression; full external PCE ablation `R^2 .713 / MAE 1.686` vs no-pretraining `.654 / 1.879` | External Lopez/Harvard computational set and Sun experimental PCE; no PCQM Gap, no closed theory/geometry contract, no verified code/checkpoint | Low-modification multi-view/electronic-objective reference; no data or weights; separate protocol only | P2 reference |
| QuADMET-Former | Public TorchMD-NET/E(3) code, HF checkpoint surface, and explicit QMugs bundle: Löwdin charge + E3FP + dipole + Gap + total energy; README reports `16/23` ADMET wins | QMugs/TPSSh-D3BJ/def2-SVP and external conformers, E3FP geometry role, ADMET downstream, no direct PCQM Gap result | Target-bundle, charge-conservation, and geometry-only control reference; no external rows or weights; separate protocol only | P1 post-selection |
| BOA | Public ICLR 2026 equivariant basis-overlap density model, code, HF checkpoint surface, and exact QM9 VASP/PySCF plus QMugs density evidence | Electron-density target, basis/grid representation, VASP/PySCF/QMugs theory and geometry, no direct PCQM Gap result | Physics-grounded electron-density teacher architecture/reference; no density rows, weights, or new database; separate protocol only | P1 post-selection |
| C-FREE | Public ICML 2026 non-contrastive ego-net objective, MIT code, HF checkpoints, and QM9 grouped HOMO/LUMO/GAP evidence | GEOM/RDKit conformers, QM9 downstream, no direct PCQM Gap result, and no current ETKDG role lineage | EMA-target/predictor objective reference; no external weights, data, or current experiment | P2 |
| OneQMC/Orbformer | Public variational wavefunction code/checkpoint, Light Atom Curriculum, NERD density extraction, and bond-breaking/Diels--Alder cost/accuracy evidence | Wavefunction/density target, LAC/NIST-derived geometry, JAX/A100 scale, light-element checkpoint, no scalar PCQM Gap result | Electronic-teacher feasibility reference; no external rows, weights, zero-shot use, or current experiment | P2/P3 |
| OrbitAll | Orbital QMM features plus explicit delta-learning; QM9star spin/charge and frontier-level evidence; OMol25-4M scale reference | spGFN1-xTB/g-xTB feature generation, B3LYP-D3(BJ)/6-311+G(d,p) and OMol25 theory/geometry, no verified code/checkpoint | Three-way delta control reference (direct/scalar low-level/orbital features); no external rows, low-level features, weights, or current experiment | P1 method reference |
| NN-xTB | Hamiltonian-preserving GFN2-xTB parameter correction, Code Ocean source/model/scripts, and energy/force/frequency benchmarks | Mixed SPICE/VQM24/DES15K/GMTKN55/rMD17 theory and geometry roles; no current HOMO/LUMO/Gap evidence; GitHub tree not yet source-complete | Low-fidelity Hamiltonian-teacher and residual-cost reference; no external model, labels, or current experiment | P2 |
| Message-Passing Delta-ML | Public ZINDO-to-TDDFT S1 residual workflow, dataset/models/tutorials, electronic descriptors, and conjugated-core split | S1/oscillator targets, ZINDO/TDDFT/external geometry roles, organic set, no PCQM Gap result | Organic-electronics Delta protocol reference; no external rows, weights, or current experiment | P2 |
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
- [new adjacent Gap-transfer readings](https://arxiv.org/abs/2607.29510): the
  NDI descriptor surrogate and ordered-to-disordered HEPO transfer study add
  narrow-domain applicability and chemistry-shift controls. They do not add
  rows, weights, geometries, or a replacement database.
- [QuantumCanvas](https://arxiv.org/abs/2512.01519), its [MIT implementation](https://github.com/KurbanIntelligenceLab/QuantumCanvas), and [Zenodo data release](https://doi.org/10.5281/zenodo.20631934) add a completed two-body electronic-teacher artifact, but the SCC-DFTB/PTBP labels, diatomic geometry, and paper/README schema discrepancy block direct use.
- [DTU solar](https://cmr.fysik.dtu.dk/solar/solar.html), [tmQM](https://github.com/uiocompcat/tmqm), and [BOS-TMC](https://doi.org/10.1021/acs.jcim.6c01792) expand the external database audit across organic donor--acceptor and transition-metal regimes. They remain theory/geometry/domain-shift references, not replacements for the current database.
- [TMC-Delta-ML](https://doi.org/10.1002/chem.71487) with [public code](https://github.com/uiocompcat/TMC-Delta-ML) closes a reproducible transition-metal Delta-learning protocol: direct versus residual modes, fixed identity splits, and cheap-versus-expensive low-fidelity graphs. Its target and geometry contract remain external to MolGap.
- [Selected Machine Learning](https://doi.org/10.1039/D2MA00742H) and its [public repository](https://github.com/b3rn4rdm/SelectedML) add a chemically interpretable family-stratification control with direct and Delta-QML scripts. The QM7/QM9 KRR evidence is useful for conditional error analysis, but it is not a PCQM/ETKDG result.
- [HLP-Stack](https://doi.org/10.1039/D5RA08007J) and its [public repository](https://github.com/college-of-pharmacy-gachon-university/HLP_STACK) add a completed descriptor-ensemble artifact, but its DFT-derived QM9 3D inputs and near-perfect metrics are a target-adjacent feature/leakage warning, not a usable current baseline.
- [PET-MAD-DOS](https://doi.org/10.1039/D5DD00557D), [Materials Cloud reproduction record](https://doi.org/10.24435/materialscloud:gs-z7), [UPET](https://github.com/lab-cosmo/upet), and [public model](https://huggingface.co/lab-cosmo/pet-mad-dos) add a complete DOS/band-gap teacher artifact with fixed-grid, mask, electron-count, UQ, and LoRA surfaces. Its PBEsol/MAD/periodic contract is external to the current B3LYP/6-31G*/ETKDG label route.
- [FieldMACE](https://doi.org/10.1038/s41524-026-02048-3), [code](https://github.com/rhyan10/FieldMACE), and [Figshare assets](https://figshare.com/articles/dataset/Models_data_and_code_for_publication_Incorporating_Long-Range_Interactions_via_the_Multipole_Expansion_into_Ground_and_Excited-State_Molecular_Simulations_/28497857) add a reproducible message-level multipole/long-range implementation and transfer controls. Its QM/MM energy/force and explicit-MM contract blocks current use.
- [MACE-POLAR-1](https://arxiv.org/html/2602.19411) and its [official foundation release](https://github.com/ACEsuit/mace-foundations/releases) add a public charge/spin-constrained electrostatic foundation checkpoint family. Its OMol25 hybrid-DFT 3D energy/force contract has no direct PCQM frontier result and remains teacher-design evidence only.
- [MACE-H](https://arxiv.org/html/2508.15108), [MIT code](https://github.com/maurergroup/MACE-H), and [Zenodo package](https://doi.org/10.5281/zenodo.15223696) add a complete operator-level KS Hamiltonian workflow. Its periodic materials/basis/SOC contract is a high-body-order reference, not molecular PCQM Gap evidence.
- [CheMeleon](https://arxiv.org/html/2506.15792), [official code](https://github.com/JacksonBurns/chemeleon), [training data](https://doi.org/10.5281/zenodo.15733574), and [released weights](https://doi.org/10.5281/zenodo.15426600) add a complete descriptor-pretrained D-MPNN transfer route. Its PubChem/Mordred/SMILES contract is external; only a train-only, same-database descriptor control could be considered later.
- [Zatom-1](https://arxiv.org/html/2602.22251), [official code](https://github.com/Zatom-AI/zatom), and [Zenodo checkpoints](https://zenodo.org/records/19766997) add a complete 3D flow-foundation and frontier-property-head artifact. Its QM9/Matbench/OMol25/MPtrj and explicit-coordinate contract remains external to PCQM4Mv2/ETKDG; only frozen-trunk/LoRA control design is portable.
- [OrbNet-Equi](https://doi.org/10.1073/pnas.2205221119) and its [Zenodo source/data/code](https://zenodo.org/records/6568437) add a direct-versus-Delta frontier-orbital control with a documented data-size crossover. Its GFN-xTB AO-feature and QM9/SDC21 contracts remain external; only the residual-control logic is portable.
- [Image-super-resolution electron density](https://doi.org/10.1038/s41467-025-60095-8), [Zenodo model/code](https://doi.org/10.5281/zenodo.15226766), and [Figshare data](https://figshare.com/articles/dataset/Image_Super-resolution_Inspired_Electron_Density_Prediction/25365508) add a complete real-space density-teacher package. Its PBE/GTH grid and one-step diagonalization route remains an electronic teacher, not a direct PCQM Gap model.
- [3DGrid-VQGAN](https://openreview.net/pdf/51c97777a512d94b366ffd7ba0d8979eba14e3c8.pdf), [IBM code](https://github.com/IBM/materials/tree/main/models/3dgrid_vqgan), and [Hugging Face checkpoint](https://huggingface.co/ibm-research/materials.3dgrid_vqgan) add a completed density-grid foundation artifact with QM9 frontier readout scripts. Its RHF/STO-3G/MINDO3 grid contract is external, and the audited `0.0088/0.0058/0.0057` table entries remain unit-ambiguous rather than eV evidence.
- [Graph2Mat](https://doi.org/10.1088/2632-2153/adc871), [code](https://github.com/BIG-MAP/graph2mat), and [DTU data](https://data.dtu.dk/articles/dataset/MD17_data_for_graph2mat/26195285) add a sparse equivariant density-matrix teacher with electron-count and self-consistency diagnostics. Its SIESTA basis/pseudopotential matrices are an operator/UQ reference, not current frontier labels.
- [ACE density matrix](https://doi.org/10.1039/D5DD00230C), [code](https://github.com/ACEsuit/ACEdensitymatrix), and [DaRUS data](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419/DARUS-4902) add Grassmann/projector-constrained density learning and commutator-residual acceptance. Its solvent/QM-MM/basis-dependent frames remain external.
- [SMILESDFT-CLIP/SigLIP](https://neurips.cc/virtual/2025/126037), [IBM code](https://github.com/IBM/materials/tree/main/models/smilesdft_clip), and [model card](https://huggingface.co/ibm-research/materials.smilesdft-clip) add a public SMILES--density contrastive pretraining and rotation-invariance control. PubChem density pairs and weights remain outside the current database and initialization boundary.
- [QMLearn](https://doi.org/10.1038/s41467-023-41953-9), [source](https://gitlab.com/pavanello-research-group/qmlearn), and [Zenodo data/code](https://doi.org/10.5281/zenodo.7946420) add a complete 1-RDM gamma/δ surrogate route with post-Fock HOMO/LUMO/Gap readouts. Its GTO/PySCF, molecule-specific, normal-mode/Eckart contract remains external.
- The [QMLearn-SCF follow-up](https://doi.org/10.1021/acs.jctc.5c01564), [open preprint](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/68c9a9763e708a76498380b5/original/main.pdf), and [Zenodo archive](https://zenodo.org/records/17103131) strengthen the residual/teacher route with SCF-threshold, force-correction, and torsion-coverage controls. Its `1.3 TB` molecule-specific GTO archive is not a current asset.
- [STRUCTURES25](https://arxiv.org/html/2503.00443v2), [official code](https://github.com/sciai-lab/structures25), and [documentation](https://sciai-lab.github.io/structures25/) add variational orbital-free density/energy supervision with perturbed-potential coverage and public inference models. It has no direct frontier head and remains a PBE/QMugs/explicit-3D teacher reference.
- [KineticNet](https://doi.org/10.1063/5.0158275) adds derivative-aware grid supervision and random-potential coverage, but its BLYP/cc-pVDZ small-system data and author-request archive keep it as a teacher-design reference.
- [M-OFDFT](https://doi.org/10.1038/s43588-024-00605-8), [Zenodo implementation](https://doi.org/10.5281/zenodo.10616893), and [Figshare model/data collection](https://doi.org/10.6084/m9.figshare.c.6877432) add the clearest audited residual-plus-gradient teacher with explicit pretraining and fine-tuning controls. Its density-coefficient and PBE/QMugs/MD17 roles remain external.
- [Meyer--Weichselbaum--Hauser](https://doi.org/10.1021/acs.jctc.0c00580) supplies a historical 1D derivative-supervision negative control: improved derivative fits do not by themselves guarantee stable unconstrained density optimization.
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
- [interaction teachers and new algorithms](deep_reading_teacher_interaction_2026-09-07.md):
  IEM supplies a same-PCQM image teacher with graph-only student inference;
  MolInteract supplies repeated 2D--3D interaction and reciprocal relation
  tasks; LeJEPA supplies a recent predictor-free pretraining control. IEM and
  MolInteract remain geometry/artifact-bounded references, while LeJEPA is an
  algorithm observation without a PCQM Gap result.
- [pretraining alignment, force objectives, and multi-domain scaling](deep_reading_pretraining_alignment_force_multidomain_2026-09-07.md):
  the NeurIPS GNN-pretraining study supplies a negative/control lesson;
  ET-OREO supplies a force-centric equilibrium/off-equilibrium objective;
  JMP supplies a public multi-domain pretraining/configuration package; and
  CSI supplies a task-aligned data-selection method with public scripts. All
  four remain bounded by external data, geometry, or downstream-task contracts;
  the only portable hypothesis is a future train-role-only same-database
  alignment test with equal-budget and scratch controls.
- Rows 114--126 now synchronize earlier deep readings that were present in the
  geometry/pretraining and teacher/delta records but missing from the unique
  ledger: GraphMVP, GeoSSL-DDM, UnifiedMolPretrain, VideoMol, PVD, D&D, CCMD,
  DelFTa, SliDe, MoleculeSDE, MoleculeJAE, MoleBlend, and FlexMol. This is an
  indexing repair, not a new database role or experiment authorization.
- Rows 127--151 synchronize the previously read multimodal, foundation,
  electronic-teacher, and delta-learning records: MolCL-SP, SpaceFormer, EMPP,
  Suiren-1.0, Uni-3DAR, C-FREE, Zatom-1, GLACIER, ChemBERTa-3, ChemFM, GPSE,
  CondPSE, embedding distillation, HEDMoL, MET, Q-GEM, overlap-population
  prediction, GW delta-learning, SelectedML, MFGP-GEM, multi-fidelity GNN
  transfer, TMC-Delta-ML, Message-Passing DeltaML, the NVIDIA ensemble, and
  DGT. These rows make existing deep-reading evidence addressable by one unique
  number; they do not authorize a warm start, data import, or experiment.
- Rows 152--165 synchronize the direct-PCQM attention and positional-encoding
  audit: local-versus-global attention, GAPE, GRPE, TetraGT, Edge-Set
  Attention, GEM-2, TokenGT, GPTrans, Edge Transformer, quantum graph
  encodings, GFSA, Specformer, AdvSynGNN, and (2,1)-GT/WL Transformers. The
  official-split, single-seed, validation-as-test, MMFF, and large-budget
  caveats remain attached to each row; none is promoted to the active queue.
- Row 166 adds GDT, a NeurIPS 2025 standard-attention/PE-control study with
  three-seed full-PCQM comparisons. Its PCQ tables are validation-style and no
  independently closed code/checkpoint or test-dev submission was found, so it
  strengthens the PE-control reference but does not authorize a run.

## Deep-reading continuation recorded on 2026-09-08

The synchronized continuation from rows 167--193 is read in
[`deep_reading_pcqm_pretraining_assets_2026-09-08.md`](deep_reading_pcqm_pretraining_assets_2026-09-08.md):

Rows 187--191 are read in
[deep_reading_pretraining_fragment_multiscale_2026-09-08.md](deep_reading_pretraining_fragment_multiscale_2026-09-08.md).

Rows 192--193 are read in
[deep_reading_teacher_geometry_2026-09-08.md](deep_reading_teacher_geometry_2026-09-08.md).

Rows 194--195 are read in
[deep_reading_sparse_probabilistic_pretraining_2026-09-08.md](deep_reading_sparse_probabilistic_pretraining_2026-09-08.md).

- **Row 167, GraphGPT.** The graph-to-sequence Eulerian serialization and
  NTP/SMTP ablations provide a real label-free pretraining design and public
  PCQM checkpoints. Its `0.0804` test-dev value is attached to a protocol in
  which 86% of validation is added to training after hyper-parameter selection,
  and the largest model is 453.4M parameters; neither is comparable to the
  sealed compact MolGap screen.
- **Row 168, 3D-GSRD.** Selective re-mask decoding and detached 3D-to-2D
  distillation are a strong leakage-control reference. The public QM9
  `homo/lumo/gap` path and PCQM pretraining recipe do not establish a direct
  PCQM Gap score or an ETKDG geometry contract.
- **Row 169, MetaGIN.** The paper supplies direct PCQM4Mv2 evidence (`0.0851`,
  `8.87M`) and a useful 1/2/3-hop path-count ablation. The “3D” interpretation
  is retained only as a static 2D topology proxy; a small path-count control is
  conditional on the existing K3 decision and is not automatically queued.
- **Row 170, EDG.** Electron-density images, an ED-aware structural teacher,
  and a frozen geometry student form the strongest electronic-teacher design in
  this batch. The Psi4, `6-31G**/+G**`, density-grid, source-conformer, and
  PCQM-derived sample roles require a new identity/theory/geometry manifest;
  no density artifact or checkpoint is imported.
- **Row 171, AUTAUT.** Automatic auxiliary-task retrieval, selection, and
  gradient-alignment weighting are read from the NeurIPS 2025 paper. Its main
  experiments are MoleculeNet, PCQM4Mv2 is only a pretrain--fine-tune source,
  and the paper-reported code URL returned 404; it is retained as an auxiliary
  supervision algorithm reference, not a current candidate.
- **Row 172, SubgDiff.** Subgraph-aware coordinate corruption combines a
  subgraph-prediction loss with expectation-state and k-step same-mask
  diffusion. Its PCQM4Mv2 role is pretraining for MoleculeNet/MD17 downstream
  evaluation, not direct PCQM Gap; the public code/data surface is useful but
  its geometry is not the ETKDG contract.
- **Row 173, UniGEM.** Two-phase nucleation/growth diffusion gates property
  supervision until the molecular scaffold forms. Public `homo/lumo/gap`
  switches and QM9 evidence make it a real loss-scheduling reference, while
  the direct PCQM role is only a Frad comparison and the main geometry/task
  contract remains external.
- **Row 174, Hyformer.** A shared Transformer alternates causal language
  modeling, bidirectional masking, and property prediction through explicit
  task masks. The paper and BSD-3-Clause repository provide reproducible
  engineering detail and public 8M/50M weights, but the pretraining corpus and
  MoleculeNet evaluations do not provide a PCQM4Mv2 HOMO/LUMO/Gap result or an
  ETKDG-compatible initialization.
- **Row 175, LAC.** Similarity-linked molecular pairs drive both a hard-sample
  curriculum and a relative edge loss; ICLR ablations show why the pair and
  node terms must be balanced. It has no direct PCQM4Mv2 Gap result or public
  implementation, so it is retained only as a bounded same-database,
  train-only loss reference pending pair-manifest and leakage controls.
- **Row 176, ST-KD.** Graph-Transformer virtual-token features and attention
  weights are transferred into a structure-tokenized SMILES student through
  feature loss and learnable attention biases. Its `0.1379` result is a
  PCQM4M-LSC validation number from the pre-v2 era, with no matched sealed
  PCQM4Mv2 role or independently retrievable official code.
- **Row 177, LW-MPP.** A later paper combines the same graph-to-SMILES
  direction with explicit attention/value relation losses and Fisher-based
  structured pruning. The paper reports three seeds, `21.5M` parameters, and
  `0.1330` historical PCQM4M-LSC validation MAE, but no code/checkpoint and no
  closed PCQM4Mv2 version/role mapping; it is compression evidence only.
- **Row 178, MolPeg.** A NeurIPS 2024 source-free data-pruning method keeps an
  online model and EMA reference model, selecting easy and hard samples by
  loss discrepancy. Its 2D GraphMAE/GraphCL encoders use PCQM4Mv2 upstream,
  but it reports downstream HIV/PCBA and QM9 results rather than PCQM Gap; it
  is therefore a train-role data-selection reference, not a current route.
- **Row 179, KPGT.** A LiGhT line-graph Transformer adds a knowledge node with
  200 RDKit descriptors and a 512-bit fingerprint, reconstructing masked graph
  nodes and masked knowledge features during roughly 2M-molecule ChEMBL29
  pretraining. Official Apache-2.0 code, public data/splits, and pretrained
  weights make it a real asset, but there is no direct PCQM Gap result and no
  ETKDG role; only a same-database auxiliary-objective adaptation is retained.
- **Row 180, KGG.** A hierarchical GIN reconstructs deterministic
  hybridization/VSEPR and bond sigma/pi/conjugation proxies, adjacency, and
  graph properties through 11 pretext tasks. Its JCIM paper, MIT code, and
  public ZINC15/ChEMBL29 model surface close the artifact gate, while the
  250K ZINC15/QM7-QM9 evidence remains indirect: these are orbital-inspired
  proxies, not actual quantum-orbital labels or a PCQM4Mv2 Gap result.
 - **Row 181, GraphFP.** Fragment-level contrastive alignment plus fragment
  existence/tree prediction uses a mined 800-fragment vocabulary (908 after
  singleton completion), 456K ChEMBL pretraining, and public code/data/weights.
  Its five-of-eight MoleculeNet result is useful structural-pretraining
  evidence, but it has no direct PCQM Gap or ETKDG result; any adaptation must
  mine the vocabulary from the permitted train role only.
- **Row 182, MoleVers.** Two-stage Uni-Mol pretraining combines masked atom
  prediction and coordinate denoising, then supervises DFT-derived HOMO, LUMO,
  and dipole auxiliaries before downstream fine-tuning. GDB17/Psi4/RDKit source
  roles, no direct PCQM4Mv2 Gap result, no verified released pretrained
  checkpoint/data package, and a paper/script noise mismatch keep it as a
  source-task pretraining reference only.
- **Row 183, supervised energy pretraining.** Gao et al. use about 86M PubChem
  PM6 optimized neutral geometries, with optional coordinate-gradient
  regularization for force tasks and scaffold-transfer controls. The route is
  useful for energy/geometry teacher design, but it has no frontier source task
  or matched PCQM/ETKDG result; PM6 rows and coordinates are not imported.
- **Row 184, PM6-ML.** The completed workflow predicts an explicit PM6-to-DFT
  energy residual with TorchMD-NET/ET, MOPAC recomputation, 21,477 molecules,
  1,145,910 conformations, 40 seeds, and a public wrapper/model surface. It is
  a strong Delta-learning artifact reference, not evidence for scalar Gap
  residuals because its target, theory, baseline, and conformer roles are
  external.
- **Row 185, O_SMI-SSM-336M.** IBM's public Mamba/SSM foundation model is
  pretrained on 91M curated PubChem SMILES and exposes code, notebooks, and
  Hugging Face weights. Its 10M-molecule HOMO/LUMO result is an inference-speed
  comparison, not a matched PCQM4Mv2 Gap result; the external sequence corpus
  and 336M scale keep it as a deployment/teacher reference.
- **Row 186, HieGT.** A direct-PCQM paper decomposes graphs into deterministic
  motifs and separates intra-motif AGA from inter-motif MGA. It reports
  `0.0769` validation and `0.0781` test-dev MAE plus a `0.0812/0.0856/0.0769`
  AGA/MGA ablation, but uses DFT training coordinates and RDKit rough
  validation/test coordinates and has no independently closed code/checkpoint.
  Keep only the bounded 2D motif-mask hypothesis.

- **Row 187, MolCHG.** A compositional hierarchy exposes atom, bond, fragment,
  and graph nodes with atom--bond contrastive alignment, fragment
  functional-group prediction, topology reconstruction, and scaffold
  prediction. The paper and public code expose exact loss weights and a
  reproducible 250K-ZINC15 recipe, but no direct PCQM Gap result; only a
  train-role-only same-database source-task adaptation is portable.
- **Row 188, BiScale-GTR.** Graph-BPE learns WL-identified and validity-filtered
  fragments, then combines an atom GINE, atom-to-fragment pooling, gated
  fusion, and a structure-aware fragment Transformer under masked-fragment
  reconstruction. Public MIT code and documented checkpoint paths close the
  mechanism/artifact lead, while ChEMBL and no PCQM/ETKDG result block direct
  transfer.
- **Row 189, FragNet.** BRICS atom, bond, fragment, and fragment-connection
  graphs provide a concrete multi-level message route and masked-fragment
  attribution. The public PNNL workflow and paper are inspectable, but
  Uni-Mol-derived pretraining and an RDKit 3D path keep the reported
  MoleculeNet evidence outside the current contract.
- **Row 190, MORE.** Four pretexts cover masked atoms, MACCS subgraphs, RDKit
  graph descriptors, and pair distances, with explicit loss weights and
  frozen-probe/fine-tune/leave-one-out controls. The paper and MIT code are
  complete enough for a control design, but ZINC2M and random/MMFF conformers
  cannot be imported; a future version would need train-role-only labels and
  ETKDG-regenerated geometry.
- **Row 191, FragmentNet.** Learned pairwise merging, WL-aware hashed
  fragments, VQVAE codes, and masked-fragment sequence modeling provide a
  non-BRICS alternative. The paper has no independently verified official
  code/checkpoint, no PCQM Gap result, and no ETKDG route, so it remains
  inspiration only.
- **Row 192, PG-MLD.** A dynamic 3D teacher combines equivariant frame
  encoding, Liquid Time-Constant sequence modeling, perturbations, and
  formal/Gasteiger charge channels before frozen atom/molecule-level
  distillation into SMILES students. The bioRxiv paper and public code close
  the method/configuration reading, but external OpenMM/ZINC roles, absent
  bundled trajectories/checkpoints, no direct PCQM Gap result, and no ETKDG
  contract keep it as a low-priority teacher reference.
- **Row 193, ConforFormer.** Conformer identity defines NT-Xent positives
  across two 3D views of the same molecule, added to Uni-Mol masked-token,
  coordinate, and distance objectives. The published paper and MIT code
  provide a clean frozen-probe/fine-tune separation and a visible pair sampler,
  but Uni-Mol/MMFF/OMol geometry and unexplained distance-normalization code
  prevent direct use; only an ETKDG train-role adaptation is portable.
- **Row 194, CardinalGraphFormer.** A query-conditioned unnormalized support
  sum is added to normalized attention over the same `K=3` shortest-path
  neighborhood, with mask-plus-contrastive pretraining and explicit size
  controls. Five-seed ADME/OGB evidence supports the mechanism, but the paper
  has no PCQM Gap task, uses external ZINC20/ChEMBL35, and its reserved
  reproducibility package was not independently retrievable.
- **Row 195, Contrastive KERMT.** Graph-to-SMILES variational reconstruction,
  cMIM in-batch mismatched negatives, and KERMT atom/bond/functional-group
  targets are written as one probabilistic objective. The combined objective
  beats the cMIM-only ablation under matched ADME protocols, but the paper has
  no PCQM Gap result, includes transductive external-corpus variants, and
  exposes only anonymized supplementary code.

This continuation broadens pretraining, teacher, and public-code coverage but
does not broaden the database. The only portable conclusions are a future
train-role-only sequence-pretraining control, a leakage-resistant teacher
manifest, a conditional small hop/path comparator, and a theory-matched
same-database electronic teacher, an alternating structure/property objective,
a train-only relative-loss audit, a deployment-only graph-to-SMILES
 compression control, a knowledge-node descriptor objective, an
 orbital-proxy auxiliary objective, a train-only fragment objective, a
 same-database frontier-source-task objective, an energy/geometry teacher gate,
  a baseline-checked scalar Delta audit, a frozen foundation-feature probe, and
  a bounded 2D motif-mask comparator, a same-database fragment source-task
  screen, a masked-fragment tokenizer audit, an ETKDG conformer-alignment
  probe, a query-gated support-cardinality channel, a probabilistic
  graph-to-SMILES/local-task pretraining objective, and a lower-priority ETKDG
  teacher/student distillation audit. All remain evidence-only hypotheses.

IPM is explicitly non-admitted: the BIBM/DBLP bibliographic records identify
the paper, but no primary full text, code, checkpoint, or data artifact was
retrieved during this audit.

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
