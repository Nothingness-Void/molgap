# Deep Reading Batch 2: Electronic Teachers and Delta Targets

Date: 2026-09-07

This batch follows the geometry/pretraining batch in
[\`deep_reading_2026-09-07.md\`](deep_reading_2026-09-07.md). It focuses on
methods that try to inject electronic information or learn a correction from a
lower-fidelity target. These papers are especially easy to misapply: a charge,
Wiberg bond order, overlap population, GW quasiparticle energy, or external
QM9 retrieval is not automatically a valid auxiliary signal for the PCQM
B3LYP/6-31G* Gap target.

The disposition in every card is evidence-only. No external database is added
to Track A or Track B, and no delta or electronic-teacher experiment is
authorized by this record.

## Evidence levels

- **A**: a primary paper reports the exact target or a directly checkable
  method result with a public artifact.
- **B**: the physics/method transfer is credible, but the target, theory level,
  database, geometry, or downstream task differs from MolGap.
- **C**: useful design evidence, but reproducibility or independent validation
  is not yet sufficient for a possible experiment.

## 1. HEDMoL: electron-informed substructure transfer

**Primary reading.** [HEDMoL paper](https://arxiv.org/html/2602.07087) and
[official code](https://github.com/ngs00/HEDMoL).

**Research question.** HEDMoL asks whether a graph representation can recover
some electron-level information for large real-world molecules without running
a new quantum calculation. Its premise is that atom-only graphs discard
information about electron density, but direct DFT/Post-Hartree--Fock
calculation is too expensive or can fail to converge for large molecules.

**Actual method.** The pipeline is explicit and reproducible in the paper:

1. Decompose an input molecular graph into atom-level substructures using a
   junction-tree decomposition.
2. Match each substructure to the most similar small molecule in an external
   calculation database using graph distance implemented with GeoScattering.
3. Transfer the small molecule’s electron-level attributes to the substructure
   graph.
4. Encode the original atom graph and the electron-derived substructure graph
   separately, then use an electron-conditioned attention readout to combine
   them.

The paper also adds an energy-consistency regularizer: the predicted energy
from atom-level and electron-derived embeddings is tied to the matched
substructure energy with an explicit approximation-noise term.

**Data and evidence boundary.** The external retrieval database is QM9. The
paper explicitly evaluates on eight experimentally collected physicochemical,
toxicity, and pharmacokinetic datasets and says it does not use simulated
datasets such as QM9 or PubChemQC as the downstream evaluation. That is useful
for the paper’s real-world-transfer claim, but it is not direct PCQM Gap
evidence. The arXiv record declares CC BY-NC-ND 4.0; the code is public, but a
license/commit/checkpoint audit is still required before reuse.

**Important negative evidence.** HEDMoL’s electron attributes are retrieved
from a nearest small-molecule match, not calculated for the target molecule.
The resulting signal can be wrong for unusual substitutions, long-range
conjugation, charge, spin, and conformers not represented in the source
database. If a downstream evaluation contains QM9 identities or close
substructures, retrieval can also create identity leakage. “No additional
calculation cost” therefore means inference-time lookup cost, not a free
electronic label.

**MolGap transfer.** **B for method, C for direct admission.** A possible
MolGap adaptation would require an external-teacher role only, canonical
identity/substructure overlap auditing, target-theory metadata, and a baseline
that sees the same official-train-derived molecules without the retrieved
attributes. It must not use QM9 retrieval as a silent feature augmentation or
as a substitute for PCQM labels.

## 2. MET: charge-supervised equivariant pretraining

**Primary reading.** [ChemRxiv MET record](https://chemrxiv.org/engage/chemrxiv/article-details/689e8887a94eede154d606f4).

**Research question.** The Molecular Equivariant Transformer (MET) asks whether
quantum-derived atomic partial charges can guide a reusable molecular
representation better than purely structural pretraining, especially in
low-data property prediction.

**Actual method.** MET combines an EGNN encoder for 3D geometry with Transformer
layers. It pretrains an auxiliary head to predict atomic partial charges, then
uses the learned representation for downstream molecular property tasks. The
paper frames the EGNN as learning symmetry-preserving spatial features and the
Transformer as adapting those features to task-specific representations.

**Evidence quality.** The accessible source is a ChemRxiv working paper, marked
as not peer reviewed at posting, with version history through August 2025. Its
abstract reports low-data gains and ablations that assign an important role to
the EGNN. The paper page uses CC BY-NC-ND 4.0. The accessible record does not
provide a PCQM4Mv2 Gap result or a verified public training repository.

**Target/geometry mismatch.** A charge is method-dependent: charge scheme,
functional, basis, geometry, total charge, and spin all matter. The method
also expects 3D coordinates, so an ETKDG-only adaptation would be different
from the reported experiment. The paper does not prove that charge supervision
improves a same-level PCQM Gap target.

**MolGap transfer.** **C/B.** Retain as a hypothesis for a named local-
electronic teacher, not as evidence for immediate execution. The first valid
step would be a source and theory audit, not a GPU screen. If a charge teacher
is later admitted, the teacher must be trained or evaluated under a frozen
geometry/theory contract and used only in the allowed train role.

## 3. Q-GEM: geometry plus CM5 charge and Wiberg bond-order supervision

**Primary reading.** [Q-GEM, Advanced Science (2025)](https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/advs.202504867).

**Research question.** Q-GEM tests whether a 3D representation becomes more
electronically informative when geometric pretraining is complemented by local
electronic targets rather than geometry alone.

**Architecture and tasks.** E-GeoGNN maintains three linked graphs:

- atom--bond graph;
- bond--angle graph;
- angle--dihedral graph.

The geometry pretraining stage masks local structures and predicts bond lengths,
bond angles, dihedral angles, and discretized all-pair distances. The electronic
stage masks atoms/bonds and predicts CM5 atomic charges and Wiberg bond orders.
The paper intentionally separates the geometry and electronic stages instead
of pretending that the two kinds of targets have the same noise model.

**Data.** Geometry pretraining uses 20 million Zinc20 molecules with RDKit
MMFF94 conformations. The electronic stage uses QuanDB, reported as 154,610
molecules with many global/local quantum properties and a lowest-energy
conformation. The paper uses DFT conformations for the electronic stage and
MMFF94 conformations for the geometry stage. These are not PCQM4Mv2 roles.

**Ablation evidence.** On MoleculeNet, Q-GEM reports state-of-the-art results
on 12/13 tasks with average improvements of about 3.3% for classification and
2.0% for regression over its comparison. More relevantly, its electronic-only
pretraining using roughly 139k QuanDB molecules is reported as competitive with
geometry pretraining using about 18M samples and better on the reported
regression tasks. A combined geometry-then-electronic schedule performs best
in the paper’s main comparison. On localized electronic properties, the paper
reports an average improvement of about 5.2%.

**Negative evidence and limits.** The paper itself notes that adding dihedrals
and electronic information yields limited gains on some downstream property
benchmarks, and it does not model long-range or intermolecular interactions.
Electronic targets are theory-specific and QuanDB is an external database.
The reported gains are not PCQM Gap gains and cannot be used to justify an
external data merge.

**MolGap transfer.** **B.** The cleanest transferable idea is a teacher-only
local electronic auxiliary target plus a geometry/electronic loss separation.
Before any possible experiment, require QuanDB metadata/license/overlap audit,
coordinate and theory normalization, and a same-PCQM/ETKDG control. Do not
import QuanDB molecules into the main PCQM or repaired-2M corpus.

## 4. Overlap-population prediction: a bond-level electronic signal

**Primary reading.** [Nakagawa et al., Chemistry Letters (2025)](https://academic.oup.com/chemlett/article/54/3/upaf038/8058640).

**Research question.** The paper asks whether a GNN can predict an overlap
population (OP) diagram directly from molecular structure, avoiding a new DFT
wavefunction calculation for every bond. OP diagrams describe bonding,
antibonding, and nonbonding orbital interactions as a function of energy, so
they are a potentially rich bond-level auxiliary signal.

**Data and model.** The authors construct a database of about 2.37 million
bonds in 130,000 QM9 molecules. Structures are calculated with FHI-aims using
PBE/GGA, collinear spin, no relativistic effects, and light numeric atom-
centered basis sets. The GNN predicts 201 evenly spaced OP values per bond. The
reported implementation uses hidden size 128, output size 256, four message-
passing blocks, a 5 Å cutoff, Adam with learning rate 1e-4, and MSE loss.

**Evidence and limitations.** The model reproduces qualitative OP trends for
unknown structures and is retrained/evaluated on larger molecules than those in
the first training subset. The paper also shows a failure mode: quantitative
peak positions and the effect of bond stretching are not always reproduced,
and the authors identify strained-molecule augmentation as a needed next step.
The article is open access under CC BY 4.0.

**MolGap transfer.** **B for feature idea, C for current teacher.** OP is a
more local bond signal than a duplicate molecular Gap label, but the source is
QM9/PBE/FHI-aims, not PCQM/B3LYP/6-31G*. It also requires a careful mapping from
the model’s bond graph to MolGap’s bond roles and would add a nontrivial
pretraining head. The correct future question is whether an OP-like bond
teacher improves an ETKDG-consistent student after identity/theory/geometry
audits; it is not a free feature import.

## 5. Delta-learning of GW quasiparticle energies from DFT

**Primary reading.** [Fediai et al., Machine Learning: Science and Technology (2023)](https://publikationen.bibliothek.kit.edu/1000163796/151619965)
and [the paper record](https://publikationen.bibliothek.kit.edu/1000163796).

**Research question.** This work studies a genuine low-fidelity-to-high-fidelity
correction: use a cheaper DFT orbital energy as the baseline and learn the
correction to expensive GW HOMO/LUMO quasiparticle energies. It is not a
HOMO--LUMO Gap architecture paper, but it is the most direct evidence in the
audited literature for the physical delta pattern around frontier orbitals.

**Actual target and baseline.** The paper uses QM9-scale data with
ev-GW@PBE calculations implemented in CP2K and aug-cc basis extrapolation for
the GW HOMO/LUMO. It studies models that predict HOMO, LUMO, and the gap, with
the Gap defined as the HOMO/LUMO difference. The learned target is the GW
correction relative to a lower-level orbital calculation, not the PCQM
B3LYP/6-31G* Kohn--Sham Gap.

**Results.** The reported GNN models reach roughly 22 meV MAE for one frontier
orbital and 31 meV for the other in the stated QM9 experiment. The work also
tests extrapolation to larger molecules and discusses simpler fingerprints as
competitive in some low-data regimes. This supports the idea that a correlated
low-level calculation can reduce the learning burden; it does not show that a
cheap proxy is correlated enough with PCQM Gap.

**MolGap transfer.** **B for the delta principle, C for direct target transfer.**
GW correction learning is scientifically valid only when the high-fidelity
target is GW. For MolGap, the same structure can be used only as a protocol
template: first compute a named low-fidelity proxy for existing PCQM molecules
under the same ETKDG coordinates, measure proxy--Gap correlation and residual
scale on the permitted internal split, and stop if coverage/cost/correlation
are unfavorable. A GW checkpoint or target cannot be substituted for PCQM
labels.

## 6. Relationship to DelFTa

[DelFTa](https://github.com/josejimenezluna/delfta) and its
[delta-QML paper](https://pubs.rsc.org/en/content/articlehtml/2022/cp/d2cp00834c)
are already covered in detail in the first batch. The additional reading here
clarifies the hierarchy:

\`\`\`text
GW/DFT delta:  same molecule, different electronic theory, target-specific
DelFTa:       cheap GFN2-xTB baseline -> higher-level QMugs target
MolGap idea:  same PCQM molecule, same ETKDG geometry, cheap proxy -> PCQM Gap
\`\`\`

Only the third line could become a MolGap experiment. It remains conditional on
the CPU residual gate and must not be confused with already closed prediction
fusion or in-sample residual routes.

## 7. Cross-paper conclusions

### 7.1 Electronic targets are not interchangeable

The cards expose four different objects:

| Signal | Typical role | Why it is not the PCQM Gap label |
|---|---|---|
| partial charge / CM5 / Fukui / NMR | local atomic teacher | charge scheme and theory are method-dependent |
| Wiberg bond order / OP diagram | local bond/electronic teacher | basis, population analysis, and geometry matter |
| HOMO--LUMO Gap | molecular target or pretraining label | functional, basis, geometry, and target role matter |
| GW quasiparticle correction | high-fidelity delta target | it predicts a different physical quantity |

An auxiliary teacher can be useful without being numerically identical to Gap,
but only if the role separation and theory metadata are explicit.

### 7.2 The strongest safe hypotheses

1. **Local electronic teacher:** predict a named atomic/bond descriptor as an
   auxiliary objective while keeping PCQM Gap as the only official target.
2. **ETKDG geometry/electronic two-stage pretraining:** use one fixed ETKDG
   construction and separate geometry from electronic proxy losses.
3. **Same-PCQM delta:** compute a cheap proxy on the existing database and learn
   the residual only after CPU correlation, coverage, and cost acceptance.

These are hypotheses, not approved runs. They must not be stacked in one
experiment because the resulting gain would be unidentifiable.

### 7.3 Reasons to reject a proposed implementation

- external QM9/QuanDB/OP labels are concatenated into the official training
  database without an overlap and role audit;
- an electronic teacher uses a different functional/basis but is described as
  “the same Gap”;
- a learned baseline produces in-sample residuals for its own training rows;
- the teacher sees DFT geometry while the student is trained or evaluated on
  ETKDG without a declared privileged-geometry contract;
- a ChemRxiv/preprint result is presented as independently replicated PCQM
  evidence;
- a low-fidelity proxy’s correlation is assumed rather than measured on the
  frozen internal split.

## 7.4 GraphQPT / atom-level quantum pretraining: useful evidence for an electronic teacher, not a PCQM Gap result

### Primary sources

- Preprint: [Analysis of Atom-level pretraining with Quantum Mechanics (QM) data for Graph Neural Networks Molecular property models](https://arxiv.org/abs/2405.14837).
- Journal version: [Pretraining graph transformers with atom-in-a-molecule quantum properties for improved ADMET modeling](https://pmc.ncbi.nlm.nih.gov/articles/PMC11869672/).
- Reproducibility code and curated files: [aidd-msca/GraphQPT](https://github.com/aidd-msca/GraphQPT).

This study is especially relevant to the requested teacher-model search because it
compares molecular-level HOMO-LUMO-gap pretraining against atom-level electronic
pretraining, rather than assuming that one scalar target is the best auxiliary
signal. It does not, however, provide evidence that an atom-level teacher improves
the current PCQM Gap task: its downstream tasks are TDC ADMET benchmarks and a
proprietary HLM clearance set.

### Method and data lineage

The backbone is a custom Graphormer with 20 hidden layers and about 10M parameters.
The centrality encoder includes explicit atoms and implicit hydrogens, while the
implementation omits a separate edge encoder. The authors compare eight settings:
scratch; four separate atom-level QM properties; a multitask model predicting all
four atom-level properties; atom masking; and graph-level HOMO-LUMO-gap (HLG)
pretraining.

The atom-level QM source contains about 136k organic molecules and more than 2M
heavy-atom labels. Its stated geometry/electronic pipeline is RDKit MMFF94s initial
conformer, GFN2-xTB optimization, then B3LYP/def2-SVP refinement. The atom-level
targets are charge, electrophilic Fukui index, nucleophilic Fukui index, and NMR
shielding. The graph-level HLG source is PCQM4Mv2, used as one scalar quantum
property per molecule. The downstream evaluation uses 22 TDC ADMET tasks with five
train/test splits; the larger HLM clearance evaluation uses a scaffold split and
three seeds.

This separation is important for MolGap: the paper's atom-level source is a
different QM protocol and geometry lineage, while its HLG pretraining consumes the
same target family as PCQM. Neither can be silently introduced into the current
random-initialized architecture screen.

### Evidence actually established

On the 22 public TDC tasks, the atom-level multitask model is reported as the best
or tied-best model on most tasks and improves over scratch on 21 of 22 tasks in the
preprint comparison. In the journal version's larger HLM experiment, the
multitask atom-level model obtains R2 values 0.640 and 0.653 for two clearance
assays, versus 0.505 and 0.534 for scratch and 0.602 and 0.607 for HLG
pretraining. The authors also report that atom-level QM pretraining produces higher
neighbor sensitivity and stronger perception of low-frequency graph-Laplacian
modes, and that this representation ranking agrees better with the large internal
dataset than the public-task ranking.

These are credible transfer and representation findings because the study includes
scratch controls, multiple pretraining types, repeated splits, and an independent
larger downstream set. They are not a direct PCQM Gap improvement, and the
proprietary HLM result is not independently reproducible from the public repository.

### What can be borrowed safely

1. Treat a local electronic multitask teacher as a falsifiable hypothesis rather
   than presuming that graph-level Gap pretraining transfers best.
2. Evaluate whether a teacher changes local-environment sensitivity, spectral
   behavior, and calibration, not only one downstream MAE.
3. Keep graph-level HLG pretraining and atom-level QM pretraining as separate
   interventions; combining them would make the source of any gain unidentifiable.
4. Require a paired scratch control and an out-of-fold teacher-output path before
   using teacher predictions as a student feature or residual input.

### Why this is not yet an experiment candidate

- The atom-level labels require an external QM cache and a MMFF94s/GFN2-xTB/
  B3LYP-def2-SVP pipeline, which is not the current ETKDG input contract.
- The PCQM HLG pretraining uses target-family labels; unless a train-only role and
  exact pretraining boundary are frozen, it can leak validation/test target
  information and cannot be called label-free.
- No paper result here measures HOMO, LUMO, or Gap on the current Track A/Track B
  split under the current inference geometry.
- The public code is a reproducibility aid, not a drop-in MolGap dependency; it
  expects its own curated datasets and an older Graphormer implementation.

Disposition: **B for electronic-teacher design evidence; C for the current screen**.
The database remains unchanged and no pretraining, auxiliary electronic-label
generation, or teacher experiment is authorized by this card.

## 7.5 Accurate GW frontier orbital energies of 134k QM9 molecules: a high-fidelity teacher database, not a B3LYP Gap replacement

### Primary evidence

- Paper: [Accurate GW frontier orbital energies of 134 kilo molecules](https://www.nature.com/articles/s41597-023-02486-4), also available as the
  [open arXiv record](https://arxiv.org/abs/2303.08708).
- Data: [Figshare dataset, DOI 10.6084/m9.figshare.21610077](https://figshare.com/articles/dataset/Accurate_GW_frontier_orbital_energies_of_134_kilo_molecules_of_the_QM9_dataset_/21610077).
- The repository is not a model implementation: the paper provides CP2K input
  information, while the generated data are distributed as a Figshare archive.

### What is actually in the dataset

The authors compute frontier-orbital energies for the 133,885 molecules of QM9.
The default eigenvalue-self-consistent GW@PBE calculation converges for 132,151
molecules; difficult cases are retried with more quadrature points or altered
CP2K settings. The calculation uses CP2K GAPW, PBE as the starting functional,
aug-cc-DZVP and aug-cc-TZVP basis sets, and an extrapolation to the basis-set
limit. The archive stores the generated records in `db_new_qm9_gw.yaml`, keyed
by the original QM9 identifiers such as `000001`, with PBE, G0W0, GW, and
auxiliary basis-set values for HOMO and LUMO.

This is valuable evidence for a teacher/data-resource search because the paper
explicitly positions the collection for HOMO/LUMO prediction, delta-learning,
and transfer learning. It is also a warning about what a teacher means: GW
frontier energies are quasiparticle quantities, not the B3LYP Kohn-Sham
HOMO/LUMO/Gap targets used by MolGap.

### Validation and limitations that matter for MolGap

The authors report large systematic theory shifts: the mean HOMO is about
−5.79 eV at PBE, −9.02 eV at G0W0@PBE, and −9.91 eV at eigenvalue-self-consistent
GW@PBE. The molecule-dependent component is not negligible. Their linear
correlations between DFT and GW are R2=0.79 for HOMO and R2=0.61 for LUMO
(G0W0 correlations are 0.90 and 0.77). The paper therefore supports a
residual-learning hypothesis, but it does not justify a fixed global correction,
especially for LUMO.

The computational cost is also evidence against regenerating such labels inside
the current project: the reported DFT+GW production consumed about 7.44 million
CPU hours and has cubic scaling in electron count. The public artifact is useful
precisely because it avoids reproducing that calculation.

The paper compares its CP2K B3LYP/6-31G(2df,p) checks with the original Gaussian
QM9 values, not with MolGap's B3LYP/6-31G* PCQM targets. It also uses QM9 IDs,
which creates a possible identity/substructure overlap path with PCQM4Mv2 and
Track A. That overlap must be measured before any representation or teacher use;
the dataset cannot be appended silently to either current role.

### Safe interpretation and disposition

- **Safe lesson:** a public, higher-level electronic teacher can be used to
  formulate a train-only residual or calibration study, with separate HOMO and
  LUMO residuals and an explicit theory/geometry audit.
- **Required gates:** canonical-identity and split-overlap audit; exact QM9 to
  PCQM mapping; target-theory declaration; train-only teacher supervision;
  out-of-fold proxy outputs; and a paired direct B3LYP-target control.
- **Not established:** improvement on current PCQM Gap, compatibility with
  ETKDG-only inference, or a valid direct teacher input for the blind test.
- **Database decision:** do not change the current database. The Figshare data
  remain an external evidence source only.

Disposition: **A for public high-fidelity DB provenance; B for delta-teacher
design; C for the current experiment screen**. No experiment is authorized by
this card.

## 7.6 VQM24: exhaustive small-molecule coverage and orbital/wavefunction assets, not a PCQM target replacement

### Primary evidence

- Paper: [A quantum mechanical dataset of 836k neutral closed-shell molecules with up to 5 heavy atoms](https://www.nature.com/articles/s41597-025-05428-4), with the [open arXiv record](https://arxiv.org/abs/2405.05961).
- Data release: [Zenodo record 15442257](https://zenodo.org/records/15442257), version 1.
- Code: [VQM24 GitHub repository](https://github.com/dkhan42/VQM24), released under MIT according to the repository metadata.
- Adapter/documentation: [OpenQDC VQM24 dataset page](https://docs.openqdc.io/stable/API/datasets/vqm24.html).

This is a finished, independently retrievable quantum-chemistry resource rather
than a model-only paper. It is relevant to the teacher/delta search because the
release contains electronic properties beyond a scalar gap, including molecular
orbital energies, multipoles, charges, and wavefunction files, while also
providing a much more systematic small-molecule coverage than QM9.

### What is actually in the dataset

The paper reports 835,947 converged structures: 784,875 local minima and 51,072
saddle points. They represent 258,242 constitutional isomers, 577,705 conformers,
5,599 stoichiometries, and neutral closed-shell molecules with at most five heavy
atoms. The element set is C, N, O, F, Si, P, S, Cl, and Br. This makes VQM24 a
coverage and chemistry-diversity resource, not a drop-in expansion of the
current PubChemQC/PCQM target population.

The geometry workflow starts from GFN2-xTB and CREST conformer sampling. The
main electronic calculations use \(\omega\)B97X-D3/cc-pVDZ with PSI4 1.7 and
frequency calculations. Diffusion Monte Carlo at PBE0 with ccECP/cc-pVQZ is
reported for 10,793 of the lowest conformers up to four heavy atoms. The paper
describes molecular orbital energies and wavefunctions among the released
electronic properties; the Zenodo record exposes a very large wavefunction
archive, while the DFT records are distributed in compressed data files.

The code release is useful for reproducibility auditing: it includes PSI4 input
templates and scripts for data/ML handling, not merely a citation or a model
checkpoint. The Zenodo page reports roughly 108 GB of listed data files, with
wavefunctions accounting for about 106.7 GB; the article reports a much larger
total data volume when all release/history components are counted. These are
asset sizes to record in a manifest, not a reason to download them into the
current project.

### What the validation actually establishes

The authors use atomization energy as a demanding ML benchmark and report that
models do not reach chemical accuracy across the VQM24 distribution, with
errors substantially larger than on QM9. This supports the use of VQM24 as a
coverage/OOD stress test: a model that looks strong on QM9-like chemistry may
not extrapolate to the exhaustive small-molecule space. It does **not** establish
that VQM24 labels improve PCQM4Mv2 HOMO/LUMO/Gap prediction, nor does it provide
a PCQM-compatible public leaderboard for the current target.

### Contract audit for MolGap

VQM24 differs from the current target contract in every field that matters for a
direct merge: the geometry path is xTB/CREST/DFT rather than ETKDG, the principal
DFT level is \(\omega\)B97X-D3/cc-pVDZ rather than B3LYP/6-31G*, the molecule
space includes elements outside the current C/N/O/F-heavy PCQM regime, and the
release contains a mixture of local minima and saddle points plus multiple
conformers per isomer. Its orbital quantities are therefore electronic-teacher
or OOD quantities, not B3LYP Kohn--Sham labels that can be appended to the
current database.

The safe identity is a constitutional-isomer/conformer record, not a SMILES-only
row. Any later use would need an exact canonical identity, charge/spin,
conformer, stationary-point status, method/basis, unit, and wavefunction-file
manifest. The DMC subset is especially unsuitable as a general training source:
it is a small, method-specific subset and is not a substitute for the current
target labels.

### Safe use and disposition

- **High-confidence use:** external coverage/OOD stratification for compact
  molecules, after defining whether the evaluation is by isomer or conformer.
- **Possible teacher use:** inspect whether orbital-energy, charge, or
  wavefunction-derived descriptors provide a useful *separate* teacher signal;
  this requires an explicit theory/geometry/identity contract and train-only
  access.
- **Not established:** any improvement on the current PCQM Gap, compatibility
  with ETKDG-only inference, or a valid direct residual target for B3LYP/6-31G*.
- **Database decision:** do not change the current database and do not download
  the large wavefunction archive as an implicit experiment.

Disposition: **A for public data/code provenance; B for coverage and teacher
design evidence; C for the current experiment screen**. No experiment is
authorized by this card.

## 7.7 QCML: massive multi-fidelity quantum data with an explicit transfer-learning hypothesis

### Primary evidence

- Paper: [The QCML dataset, Quantum chemistry reference data from 33.5M DFT and 14.7B semi-empirical calculations](https://www.nature.com/articles/s41597-025-04720-7).
- Versioned metadata/examples: [Zenodo record 14859804](https://zenodo.org/records/14859804).
- Data: the paper and Zenodo record expose the public [Google Cloud TFDS bucket](https://console.cloud.google.com/storage/browser/qcml-datasets/tfds/).

QCML is a completed public data resource with unusually explicit hierarchy and
failure handling. It starts from 17.2 million chemical graphs, creates
14.678 billion conformations/results with GFN0-xTB and GFN2-xTB, and calculates
a randomly selected 33.496 million subset at PBE0/FHI-aims. The article frames
the resource as a foundation for force fields and transfer learning, not as a
HOMO/LUMO benchmark.

### Data contract and reproducibility

The hierarchy is chemical graph -> conformer -> quantum calculation. The graph
space is restricted to at most eight heavy atoms for downstream processing,
while the element set covers a large fraction of the periodic table and the
electronic state includes charge and inferred multiplicity. Coordinates are
created with Open Babel, UFF pre-optimization, GFN0-xTB BFGS optimization,
Open Babel genetic conformer search, symmetry-aware RMSD filtering, and
100--1,000 normal-mode samples per conformer over a 0--1,000 K energy range.
This is a deliberately off-equilibrium dataset rather than an equilibrium-only
molecule table.

The semi-empirical collection includes energies, forces, Wiberg/Mayer bond
orders, charges, orbital occupations, and orbital energies. The DFT collection
uses PBE0 with FHI-aims default tight numeric atom-centered basis settings and
also supplies PBE0 matrices/densities, multipoles, charges, and dispersion
corrections. The public record makes the storage scale auditable: the xTB
collection is listed at about 69 TB, while individual PBE0 matrix/grid
collections are listed at tens of TB. The released example scripts and TFDS
metadata are more useful to MolGap than attempting a bulk download.

The authors apply graph sanitization, convergence filters, bond-order checks,
stationary-point checks, and an explicit is_outlier flag. About 1.5% of the
entries are flagged by their formation-energy, force, distance, or bond-order
criteria. This is valuable engineering evidence: a future teacher cache needs
status and outlier fields, rather than silently treating every generated
coordinate as equally valid.

### What the paper actually demonstrates

The paper's validation trains SpookyNet to predict PBE0 formation energies and
forces, with three replicate runs at increasing data scales up to 30M examples.
Energy and force errors fall below chemical-accuracy thresholds around one
million examples and continue to improve before saturating between 10M and 30M.
For the multi-fidelity question, the authors show a high correlation between
GFN2 and PBE0 formation energies with a systematic error, and show that
differences between samples of the same molecule can reduce that systematic
component. They explicitly present transfer learning from the 14.7B xTB pool
as a hypothesis; this is not evidence for a PCQM HOMO/LUMO/Gap gain.

### MolGap contract audit

QCML does not match the current target in geometry, theory, or task: its
coordinates are UFF/GFN0-xTB/conformer-search/normal-mode based rather than
ETKDG; its high-level labels are PBE0/FHI-aims energies, forces, and electronic
matrices rather than B3LYP/6-31G* Kohn--Sham HOMO/LUMO/Gap; and its off-equilibrium
sampling, charged/open-shell states, and broad elements require a different
input contract. The xTB orbital-energy fields are not interchangeable with the
current target either.

**Safe use:** retain the hierarchical schema, convergence/outlier manifest,
same-graph low/high-fidelity residual principle, and compute-time accounting as
design evidence. A future compact teacher could use a deliberately selected,
identity-controlled subset for electronic pretraining, but only with an
explicit theory/geometry role and a paired current-target control.

**Database decision:** do not merge QCML into Track A or Track B, and do not
download its multi-terabyte collections as an implicit experiment.

Disposition: **A for public data/provenance; B for multi-fidelity pretraining
and residual design; C for the current PCQM experiment screen**. No experiment
is authorized by this card.

## 7.8 qcMol: near-target electronic descriptors at larger, real-world chemical scale

### Primary evidence

- Paper: [A dataset of 1.2 million molecules with DFT-level quantum chemical annotations for molecular representation learning](https://www.nature.com/articles/s42004-026-02076-6).
- Public implementation and pipeline: [qcMol GitHub repository](https://github.com/GHUSER-haoyu/qcMol).
- Data/parameters: the paper links the [qcMol web server](https://structpred.life.tsinghua.edu.cn/qcmol/), [benchmark data](https://zenodo.org/records/19957978), and [parameter archive](https://zenodo.org/records/19183364).

qcMol is particularly relevant to an electronic-teacher search because it
contains 1,200,216 curated molecules, 247,448 scaffold types, and 31 quantum
descriptors (15 global and 16 local). The global set includes HOMO--LUMO Gap;
the local features are obtained from NBO/Multiwfn wavefunction analysis and
include atom- and bond-level electronic descriptors. About 63.65% of the
molecules are associated with systematic experimental measurements, but those
experimental labels are not the MolGap target and are not needed for the
quantum-teacher case.

### Calculation and training evidence

The calculation contract is B3LYP-D3/def2-SV(P)//GFN2-xTB using RDKit/Open
Babel for initialization, xTB ANCopt with stringent convergence settings, ORCA
5.0.3 for DFT, and NBO 7.0/Multiwfn 3.8 for wavefunction analysis. The data
are predominantly neutral singlets, with reported doublet and charged
fractions. Each molecule is represented by one optimized geometry; the authors
explicitly state that exhaustive conformer search was not performed.

For representation evidence, the authors pretrain a six-layer SchNet on
50,000-molecule subsets of qcMol and QM9 using the same masked-reconstruction
setup, then evaluate frozen and fully fine-tuned models on downstream ADMET
tasks. Every configuration is repeated over ten random seeds. qcMol
pretraining wins in nearly all reported settings, but the paper also reports
that full fine-tuning can largely erase the advantage on some data-rich tasks.
The paper's result is therefore evidence that data distribution and local
electronic descriptors can help representation learning; it is not a direct
PCQM4Mv2 Gap result.

### MolGap contract audit

The apparent proximity of the name “B3LYP” is not enough for label reuse.
qcMol adds D3 dispersion, uses def2-SV(P) rather than 6-31G*, and evaluates
single points on GFN2-xTB-optimized rather than ETKDG coordinates. It also
deduplicates at its own molecule/conformer policy and aggregates 95 source
datasets, so overlap with PubChem/PCQM and with any future teacher split must
be measured from canonical identity and source IDs. A single geometry per
molecule is not an ensemble substitute for the current ETKDG inference path.

**Safe use:** qcMol is a strong post-selection candidate for an external,
identity-filtered electronic representation or teacher audit, especially for
atom/bond descriptors and size/scaffold stress tests. The audit must preserve
functional, basis, geometry, charge/spin, source, and descriptor provenance;
it must not use the experimental ADMET labels as an unannounced auxiliary task.

**Database decision:** no qcMol rows are added to the current database and no
pretraining is authorized by this card.

Disposition: **A for public dataset/code provenance; B for near-target
electronic-teacher design; C for the current screen**.

## 7.9 QM40: B3LYP frontier-orbital labels beyond QM9, with a different basis and geometry role

### Primary evidence

- Paper: [QM40, Realistic Quantum Mechanical Dataset for Machine Learning in Molecular Science](https://www.nature.com/articles/s41597-024-04206-y).
- Data and code: [QM40 Figshare record](https://doi.org/10.6084/m9.figshare.25993060.v1) and [QM40_dataset_for_ML repository](https://github.com/Ayeshmadu/QM40_dataset_for_ML).
- The paper's data table identifies HOMO, LUMO, and HL_gap among the main
  quantum parameters; the repository provides download and extraction tooling.

QM40 contains 162,954 neutral, singlet, drug-like molecules with 10--40 atoms
and C, N, O, S, F, and Cl. The selection is from ZINC and is designed to
represent 88% of the authors' FDA-approved-drug chemical-space analysis. The
main dataset includes 16 quantum parameters, and the release also contains
initial/optimized coordinates, Mulliken charges, and local vibrational-mode
bond-strength values.

All reported quantum calculations use Gaussian16 at B3LYP/6-31G(2df,p), with
initial structures prepared through RDKit and GFN2-xTB, followed by DFT
optimization and frequency analysis. The paper reports automated rejection of
convergence failures, imaginary frequencies, and connectivity/force-constant
inconsistencies. The code repository is inspectable and MIT-labeled; the paper
and repository describe the surrounding data/code permissions differently, so
any later reuse requires freezing the exact data license and release.

### MolGap contract audit

QM40 is closer to the current electronic target family than QCML or VQM24
because it exposes HOMO/LUMO/Gap at B3LYP, but it still is not a current-label
replacement: 6-31G(2df,p) is not 6-31G*, the geometries are DFT/xTB-derived
rather than ETKDG, the chemical distribution is ZINC drug-like rather than the
official PCQM split, and charged/open-shell cases are excluded. Exact canonical
identity and source overlap with PCQM/Track A are unknown until the files are
audited.

**Safe use:** external size/scaffold stress testing and a possible
identity-filtered teacher/representation audit after license and target
mapping. The local bond-strength features are a separate auxiliary family, not
proof of Gap improvement.

**Database decision:** do not concatenate QM40 with the current database and do
not call its Gap a PCQM label.

Disposition: **A for data/code provenance; B for near-target external
validation and teacher design; C for the active PCQM screen**.

## 7.10 ESA frontier-orbital transfer: a direct DFT-to-GW control with inductive and transductive splits

### Primary evidence

- Paper: [An end-to-end attention-based approach for learning on graphs](https://www.nature.com/articles/s41467-025-60252-z), Nature Communications 16, 5244 (2025); the [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12141427/) exposes the transfer-learning table.
- Code: [official edge-set-attention repository](https://github.com/davidbuterez/edge-set-attention), which includes the `transfer_learning` path and 3D adaptations.
- High-fidelity source: the same [134k QM9 GW frontier-orbital release](https://figshare.com/articles/dataset/Accurate_GW_frontier_orbital_energies_of_134_kilo_molecules_of_the_QM9_dataset_/21610077) audited above.

This paper contributes an independent transfer-learning control for frontier
orbitals. It uses a refined QM9 DFT/GW collection, a 25k/5k/10k high-fidelity
train/validation/test split, and separate low-fidelity DFT models. In the
**transductive** setting, the low-fidelity model sees the full molecular set; in
the **inductive** setting, the high-fidelity test molecules are removed from
low-fidelity training. This distinction is essential: a transfer gain from
seeing the test structures at low fidelity is not the same claim as a gain on
unseen molecules.

The model is ESA, an edge-set attention encoder with masked attention over
incident edges, ordinary self-attention, and attention pooling. The transfer
experiment uses only atom types and 3D coordinates, with no extra node or edge
features, and reports RMSE averaged over five runs. Without transfer, ESA and
PNA are close; after transfer, ESA is best or tied on both HOMO and LUMO in the
reported inductive and transductive comparisons. For example, the table gives
ESA HOMO RMSE 0.152 for GW-only, 0.131 for inductive transfer, and 0.119 for
transductive transfer; the corresponding PNA values are 0.151, 0.132, and
0.121. For LUMO, ESA is 0.174 for GW-only, 0.150 for inductive transfer, and
approximately 0.146 for transductive transfer; these values should be read as
the paper's QM9/GW RMSE, not PCQM Gap MAE.

The repository is unusually transparent about a separate reproducibility
limitation: for PCQM4Mv2 it uses the public validation set as a test surrogate
because the official test labels are not public, and its README says early
stopping must be disabled for that route. Therefore the paper's direct PCQM
validation number is not admitted as a leaderboard claim here. The frontier-
orbital transfer result is stronger evidence because it has a named high-
fidelity split, five runs, and explicit inductive/transductive definitions.

### MolGap judgment

This is evidence for three practices that should be preserved in any future
same-database teacher/delta protocol: (1) separate inductive from transductive
transfer, (2) keep the low-fidelity source molecule audit explicit, and (3)
compare transfer against a high-fidelity scratch control with repeated runs.
It does not authorize importing the QM9 GW labels, because the theory is GW
quasiparticle energy rather than B3LYP/6-31G* Kohn--Sham energy and the source
may overlap with PCQM/Track A. It also does not establish a current ETKDG
result: the transfer study uses QM9 3D coordinates and a different target
database.

Disposition: **A for frontier-orbital transfer protocol and split hygiene; B/C
for current MolGap use.** No data, checkpoint, or experiment is added.

## 7.11 MFGP-GEM: multi-fidelity autoregression with a HOMO/LUMO benchmark

### Primary evidence

- Paper: [Rapid high-fidelity quantum simulations using multi-step nonlinear autoregression and graph embeddings](https://www.nature.com/articles/s41524-024-01479-0), npj Computational Materials 11, 57 (2025); the [published PDF](https://www.nature.com/articles/s41524-024-01479-0.pdf) contains the method and implementation details.
- Code and benchmark data: [official MFGP-GEM repository](https://github.com/ashah1973/MFGP-GEM/tree/main).

MFGP-GEM is a more explicit multi-fidelity alternative to a one-step residual
head. It first builds a molecular dual-graph spectral embedding using diffusion
maps, then embeds the molecules again in a molecule-level graph, and finally
feeds the resulting representation into a nonlinear multi-step autoregressive
Gaussian process. The autoregressive inputs can include several lower/medium
fidelities, rather than assuming that one scalar low-level prediction is enough.
The paper compares it with direct GP/GCN/MEGNET, linear and nonlinear
autoregression, stochastic collocation, MF-MEGNET, CQML, and delta-ML.

The closest benchmark is a new public set of 860 benzoquinone molecules with up
to 14 atoms. It contains energy, HOMO, LUMO, and dipole values at four levels:
HF, B3LYP, MP2, and CCSD, all with a cc-pVDZ basis. The structures are generated
from SMILES with RDKit; the paper's methods state that the quantum calculations
use optimized geometries based on unrestricted HF. This is a small,
domain-specific benchmark, not a replacement for PCQM4Mv2.

Across five benchmark problems, 14 quantities, and 27 theory levels, the paper
reports that MFGP-GEM generally reaches useful high-fidelity accuracy with a few
tens to a few thousands of high-fidelity points, often far fewer than direct ML
and fewer than the compared multi-fidelity baselines. The quinone experiments
also show why HOMO/LUMO should not be treated as an ordinary total-energy
correction: the paper notes that orbital quantities are harder when the
low/high-level theories treat electron correlation differently. The public code
and the paper's released quinone data make this a real method/data artifact,
not a README-only proposal.

### MolGap judgment

The transferable lesson is to test a hierarchy of fidelities and to plot error
against both high-fidelity sample count and label-generation cost. A same-row
PCQM delta study could borrow the ablation logic: direct high-level model,
one-step residual, and multi-step/autoregressive correction, all with a cheap
proxy that is available at deployment. The paper also provides a useful negative
check: when the low/high theories are weakly correlated for a frontier orbital,
delta learning may not compress the problem.

The benchmark cannot be imported into MolGap. It uses a narrow benzoquinone
family, cc-pVDZ, multiple theories, and a geometry/calculation workflow that is
not the B3LYP/6-31G* plus ETKDG contract. Running HF/MP2/CCSD or appending the
860 rows would change the database and target role. The current safe action is
only to retain the model-selection and cost-accounting protocol.

Disposition: **A for multi-fidelity/Δ-learning methodology and an auditable
HOMO/LUMO benchmark; C for current PCQM use.** No data or experiment is added.

## 7.12 Multi-fidelity GNN transfer: labels, predictions, embeddings, and readout-only tuning

### Primary evidence

- Paper: [Transfer learning with graph neural networks for improved molecular property prediction in the multi-fidelity setting](https://www.nature.com/articles/s41467-024-45566-8), Nature Communications 15, 1517 (2024); the [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC11258334/) provides the detailed strategy comparison.
- Code: [official multi-fidelity GNN repository](https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics), including QMugs assembly indices and training scripts.

This is one of the most useful protocol papers for the current teacher/delta
question because it does not collapse all transfer into "pretrain then fine-
tune." It evaluates six distinct ways to inject low-fidelity information into
a high-fidelity model:

1. explicit low-fidelity labels;
2. low-fidelity predictions from a separately trained model;
3. a hybrid that trains with raw low-fidelity labels but uses predictions at
   inference;
4. low-fidelity latent embeddings with either sum or neural readout;
5. ordinary low-fidelity pretraining followed by high-fidelity fine-tuning; and
6. fine-tuning only an adaptive readout while freezing the representation.

The neural readout is a permutation-invariant Set Transformer-style adaptive
readout over node states. This matters for electronic quantities because a fixed
sum/mean readout may discard which local environments are responsible for the
property. The code includes both the paper's multi-fidelity state-embedding
baseline and the neural/sum alternatives, with a QMugs data-assembly path.

### Quantum-mechanics evidence

The quantum study uses 12 QMugs properties, including HOMO and LUMO, with
GFN2-xTB as a low-fidelity source and DFT properties as the higher-fidelity
source. After filtering, the article describes 647,794 QMugs examples for the
main low-fidelity analysis and also evaluates a challenging 10K diverse subset.
Both transductive and inductive settings are tested. In the transductive case,
low-fidelity information can be available for all molecules, including the
high-fidelity test identities; in the inductive case, the high-fidelity test
molecules are removed from low-fidelity training. This is a critical separation
for any MolGap audit.

The paper reports that adaptive neural embeddings frequently outperform raw
labels and sum embeddings. In its count-of-best summary across the QMugs and
drug-discovery groups, the neural strategies win all 40 counted cases, with
readout tuning the most frequent winner (26 cases). For individual QMugs
properties, the transductive MAE is nearly halved for LUMO and is reduced by
roughly 4--8 times for several total/atomic-energy and rotational-constant
tasks in the 8K high-fidelity regime. The paper also extends the strategy to
three fidelities (ZINDO, PBE0, and GW HOMO/LUMO in QM7b), rather than assuming
that only two levels exist.

These gains are not all equivalent. A low-fidelity label available for every
test molecule is a transductive feature; it can be valid in a deployment funnel
but must not be described as inductive generalization. A frozen-backbone,
readout-only gain is a different claim from a full-network fine-tuning gain.
The paper keeps these comparisons separate, which is stronger evidence than a
single mixed-fidelity score.

### MolGap judgment

The directly reusable part is the experiment matrix, not the QMugs labels:

- direct high-fidelity control;
- a train-only low-fidelity proxy audit;
- proxy-label, predicted-proxy, proxy-embedding, full fine-tune, and readout-only
  variants;
- separate inductive and deployment-valid transductive reports; and
- per-target results, not only one aggregate metric.

This work also strengthens the current gate that a cheap proxy must be available
for every inference molecule. If it is not, a proxy feature that helped on an
offline transductive split is not a deployable MolGap feature. The project must
also retain the exact low/high theory, units, charge/spin, geometry, and failure
policy, because QMugs' GFN2-xTB/DFT contract is not B3LYP/6-31G* on the ETKDG
path.

No QMugs row, checkpoint, or low-fidelity calculation is imported. A future
same-database screen would require explicit authorization after the current
architecture screen, and it would need to compare readout-only transfer against
an identical fresh direct control. The paper's public code is an implementation
reference, not a permission to reuse its external data.

Disposition: **A for multi-fidelity strategy design, leakage/split hygiene, and
readout-only evidence; B/C for current MolGap use because of the database,
geometry, and theory mismatch.**

## 7.13 Trainable dataset embeddings: fidelity conditioning without pretending labels are identical

### Primary evidence

- Paper: [Multi-fidelity learning for atomistic models via trainable data embeddings](https://doi.org/10.1088/2632-2153/ae0d41), Machine Learning: Science and Technology 6, 045004 (2025).
- Code/model: [Fraunhofer-SCAI VMDatomistic](https://github.com/Fraunhofer-SCAI/VMDatomistic), released with an MIT license according to the repository.
- Public sources named by the paper: MultiXC-QM9 and MatPES; the released repository exposes a multi-fidelity M3GNet evaluation path.

This work addresses a different failure mode from ordinary delta-learning: the
training pool contains labels generated by different functionals and basis sets,
so the values are not interchangeable. The model learns a trainable embedding
vector for each reference dataset/fidelity and feeds it into the atom-pair
embedding function. A shared M3GNet-like backbone can therefore represent
method-dependent outputs while sharing chemical environment information.

The experiments cover two scenarios. In one, a large dataset is available at a
cheap level and a small subset is re-labelled at an expensive level. In the
other, several datasets with different DFT methods are jointly available. The
paper uses MultiXC-QM9 for multiple functional/basis combinations and MatPES for
PBE/r2SCAN M3GNet tests. Ten DFT method combinations are jointly evaluated with
10,000 training points per method and ten resampling runs. The article reports a
large reduction in expensive-label demand (including a factor-of-16 reduction
for an M06-2X scenario); the public model description separately reports a
factor-of-10 reduction for the PBE/r2SCAN MatPES case. These are energy/force
model results, not HOMO/LUMO/Gap results.

### MolGap judgment

The transferable safety lesson is valuable: if heterogeneous quantum labels are
ever used together, the method, basis, geometry, and state must be an explicit
conditioning variable or a separate role. A single target column cannot be
made consistent by renaming it. Dataset/fidelity embeddings could be a future
multi-task design reference, but they would not remove the need for target-theory
and identity audits.

The present project has one declared B3LYP/6-31G* target family and an ETKDG
train/inference geometry contract. No external functional/basis labels, MatPES
rows, or VMDatomistic checkpoint is imported, and no fidelity-conditioned model
is authorized by this card.

Disposition: **A for explicit fidelity conditioning and heterogeneous-label
handling; C for current HOMO/LUMO/Gap performance.**

## 7.14 PubChemQC B3LYP/6-31G*//PM6: same-origin electronic labels, not a silent database extension

### Primary evidence

- [Official PubChemQC project index](https://nakatamaho.riken.jp/pubchemqc.riken.jp/)
  lists the 2017 B3LYP release, the PM6 release, and the 86-million-molecule
  B3LYP/6-31G*//PM6 release as separate products.
- [The 86M dataset paper](https://arxiv.org/abs/2305.18454) reports
  85,938,443 molecules, B3LYP/6-31G* electronic calculations on PM6-optimized
  geometries, and orbital energies, total energies, dipoles, and related
  properties. It provides GAMESS files, selected JSON, and a PostgreSQL form;
  the project page links the release under CC BY 4.0.
- The official [2017 database archive](https://nakatamaho.riken.jp/pubchemqc.riken.jp/b3lyp_2017.html)
  publishes checksums and an archived PostgreSQL/PostgREST image. Its query
  schema exposes alpha/beta HOMO, LUMO, and Gap indexes.

This is the closest public source found to the project's electronic target
lineage. The 86M paper explicitly distinguishes the calculation level from the
geometry level: `B3LYP/6-31G*//PM6` means that the geometry was optimized with
PM6 and the electronic structure was then evaluated with B3LYP/6-31G*. Across
its listed subsets, the paper reports HOMO--LUMO-gap correlations of roughly
`R^2 = 0.803--0.892` between PM6//PM6 and B3LYP/6-31G*//PM6 on the same
geometries. That correlation is useful evidence for a low/high relationship,
but it is not evidence that those values equal the PCQM4Mv2 labels or the
repaired-2M labels under the MolGap ETKDG contract.

### Identity and contract boundary

The release is a broad PubChem snapshot (retrieved in 2016), while PCQM4Mv2
is a fixed OGB release and Track A is the existing repaired-2M corpus. Exact
CID/InChI overlap, charge/spin and failure coverage, label convention, and
coordinate identity have not been audited here. PM6 coordinates also cannot
be used as ETKDG training or inference coordinates. Therefore the source must
not be concatenated, used to add HOMO/LUMO labels, or used to initialize a
current architecture screen.

### MolGap judgment

The only defensible future role is a separately authorized same-origin
electronic-teacher or low/high-proxy study. It would first need immutable
release hashes, exact identity mapping against every current role, a
train-only overlap report, a target-theory comparison, and an explicit choice
between (a) using only electronic values as a teacher target or (b) recomputing
an allowed proxy on the existing ETKDG molecules. In either case the main
database remains unchanged.

Disposition: **A for source/data-lineage discovery; B for a possible
same-origin teacher or delta design; C for current training use.**

## 7.15 MFΔML: a directly auditable delta-learning cost rule

### Primary evidence

- [Vinod and Zaspel, JCP 2025 / arXiv:2410.11391](https://arxiv.org/html/2410.11391)
  introduces Multifidelity-Delta-ML (MFΔML) and reports learning curves and
  compute-cost accounting on QeMFi.
- The authors release [MFDeltaML code](https://github.com/SM4DA/MFDeltaML),
  which contains separate scripts for Δ-ML, MFML, optimized MFML, MFΔML and
  the predicted-baseline variant. The README points to a
  [QeMFi Zenodo snapshot](https://zenodo.org/records/12734761); the QeMFi
  paper's later data DOI is [10.5281/zenodo.13925688](https://doi.org/10.5281/zenodo.13925688).
  Those release identifiers must be frozen before any reproduction.

QeMFi supplies 135,000 geometries of nine molecules at five nested basis-set
levels. Standard Δ-ML learns a residual `y_high - y_base` and adds a fresh
low-fidelity calculation at prediction time. MFML predicts the low-fidelity
components instead, avoiding that per-query quantum-chemistry cost. MFΔML
centers the several fidelity targets on a lowest-fidelity calculation and
combines multiple residual models. This makes the distinction between a
residual model and a multi-fidelity ensemble explicit rather than treating both
as generic ``delta learning''.

The paper reports, at a fixed 2^11 highest-fidelity training size, the following
MAEs: ground-state energy `11.41/2.93/2.54/1.30` for KRR/MFML/Δ-ML/MFΔML;
first excitation `2.54/1.42/1.72/1.18`; second excitation
`2.69/1.92/2.60/1.93`; and electronic dipole magnitude
`0.16/0.07/0.06/0.04` in their respective units. The more important result
for MolGap is the cost curve: with an 80K test set, the cost of computing the
baseline for every query dominates Δ-ML, so MFML is preferred for many
predictions; MFΔML remains more attractive than ordinary Δ-ML when only a
small number of evaluations is needed. The supplementary predicted-baseline
variant also shows that baseline-prediction error can dominate and remove the
expected benefit.

### MolGap judgment

This supplies a reliable protocol rule for any future same-database delta
study: report residual variance and high/low correlation, but also report
inference-time proxy cost and the number of required proxy evaluations. A
proxy that is itself learned must be out-of-fold for residual construction and
must be compared with a direct model under the same ETKDG input. If the proxy
is unavailable for a blind molecule, the route is not deployment-valid.

QeMFi's nine molecules, basis-set fidelities, Coulomb-matrix descriptor, and
TD-DFT geometry contract are not the current PCQM/Track A task. No QeMFi row,
code, or checkpoint is imported, and no MFΔML experiment is authorized.

Disposition: **A for delta-cost and nested-fidelity methodology; B for a
future same-database residual protocol; C for current target performance.**

## 7.16 High-throughput GFN2-xTB gap asset for proxy and OOD audits

### Primary evidence

[Thinius, Digital Discovery 2026](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00186b)
uses `407,000` natural products from COCONUT to construct a public low-fidelity
HOMO--LUMO-gap resource. The workflow generates ten RDKit conformers per
molecule, optimizes each with GFN2-xTB/BFGS, calculates GFN2-xTB orbital gaps,
and combines them with Boltzmann weights. The target is explicitly calculated
for this study rather than inherited from COCONUT. The reported split is `70/30`
train/test, with repeated shuffle-split evaluation for the selected regressors.

The article reports `0.210 +/- 0.001 eV` test MAE and `0.298 +/- 0.002 eV`
RMSE for its more stable MLPR model; XGBoost reaches a lower single-split MAE
of about `0.180 eV` but has a much larger train-to-test generalization gap. The
authors also evaluate the model on approximately `133,000` unseen QM9
molecules against B3LYP/6-31G(2df,p) and GW references. The xTB-trained model
shows a systematic theory shift rather than a same-label prediction: the paper
reports mean underestimation of about `3.94 eV` relative to the QM9 DFT gaps and
`7.75 eV` relative to GW gaps.

### Public asset and what it proves

The paper states that the [GitHub workflow and full calculated
dataset](https://github.com/sthinius87/HL-gaps-pub) are available as version
`v0.2.1`, with a persistent [Zenodo archive](https://doi.org/10.5281/zenodo.15113790).
The article states MIT licensing for the code and data; the upstream COCONUT
source remains a separate provenance/redistribution question. This is unusually
useful for the project because it exposes a complete low-fidelity workflow,
hyperparameters, descriptors, error subgroups, and archived artifacts rather
than only a headline score.

### MolGap contract audit

This is not a B3LYP/6-31G* teacher. Its geometries are RDKit-generated and
GFN2-xTB optimized, while MolGap requires ETKDG at both training and inference;
its target is GFN2-xTB and its chemical distribution is natural products, not
the official PCQM4Mv2/Track A role. A direct copy would therefore mix geometry
and theory. The safe transferable lesson is operational: a proxy audit should
record the low-fidelity method, conformer policy, aggregation rule, feature
cost, uncertainty/error strata, and out-of-distribution theory shift. A future
same-PCQM residual study could borrow the workflow shape only after computing a
proxy under the exact ETKDG contract and measuring correlation/residual variance
on the frozen internal split.

**Disposition: A for public low-fidelity workflow/data provenance; B for a
future delta-protocol reference; C for current MolGap use.** Do not merge
COCONUT/xTB rows, use the xTB weights as a current teacher, or report its MAE as
PCQM evidence.

## 7.17 QMCVNet: same-lineage low-fidelity geometry control

### Primary evidence

[Maser and Reisman, “3D Computer Vision Models Predict DFT-Level HOMO-LUMO Gap
Energies from Force-Field-Optimized Geometries”](https://authors.library.caltech.edu/records/2jygg-n1r30)
is a 2021 ChemRxiv discussion paper with an attached [paper PDF](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60ea947a9ab06e2e274d6cd7/original/3d-computer-vision-models-predict-dft-level-homo-lumo-gap-energies-from-force-field-optimized-geometries.pdf)
and [supplementary information](https://authors.library.caltech.edu/records/2jygg-n1r30).
The Caltech record identifies the work as submitted material under CC BY-NC-ND
4.0, not a peer-reviewed production code release.

The study uses a PubChemQC PM6 collection of more than `200M` molecules and a
`2.5M`-structure subset with B3LYP/6-31G* calculations. It retains about
`1.8M` molecules after element/molecular-weight filtering and outlier handling
and predicts the B3LYP HOMO--LUMO gap from voxelized 3D structures. The input
geometry is not one fixed contract: the authors use PM6 geometries or RDKit
MMFF geometries, align MMFF structures to PM6 structures, and evaluate a random
`100k` subset with an `80/10/10` split before scaling to `1M` structures.

The main QMCVNet has `8.7M` parameters; ShapeEncoder-d is larger than `15M`.
On the `100k` screen, the PM6 gap baseline is `0.400 eV` MAE, while
ShapeEncoder-d reaches `0.539 eV` from PM6 coordinates and QMCVNet reaches
`0.549 eV` from MMFF coordinates. Ten-fold right-angle rotation augmentation
brings ShapeEncoder-d/MMFF to `0.502 eV`. On `1M` structures, ShapeEncoder-d
reports `0.418 eV` from PM6 coordinates after 20 epochs and `0.455 eV` from
MMFF after 100 epochs. These numbers show a real low-fidelity/geometry
comparison, but they do not establish a strict residual model or a gain over
the explicit PM6 electronic baseline.

### MolGap contract audit

The work is useful because it makes three controls explicit: low-fidelity
geometry versus a force-field geometry, a same-lineage B3LYP target, and
rotation augmentation cost. It is not legal evidence for a current MolGap
candidate: PM6/MMFF coordinates are mixed and aligned, the data lineage may
overlap Track A, the representation is a voxel CNN rather than GraphState, the
paper is a discussion preprint, and no official code/checkpoint is exposed in
the audited record. Its “delta” framing is also not strict residual learning;
the neural model predicts the high-level gap directly from geometry rather than
publishing `y_high - y_low` as the supervised target.

**Disposition: B for a historical same-lineage low-fidelity/rotation-control
reference; C for current initialization, labels, or database import.** If a
future proxy study is ever reopened, the only transferable design is the
paired direct-versus-proxy control and explicit rotation/geometry accounting,
recomputed from ETKDG under the project contract. Do not use PM6 coordinates as
training inputs for an ETKDG inference path.

## 7.18 QUED: reproducible electronic-descriptor feature path

### Primary evidence

[Hinostroza Caldas et al., Digital Discovery 2026](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00411j)
introduces QUED, a hybrid representation that combines inexpensive geometric
descriptors (BOB or SLATM) with a DFTB3+MBD electronic descriptor `D_QM`. The
descriptor contains global quantities, molecular-orbital energies, and padded
atomic properties such as Mulliken charges. The paper evaluates approximately
`42k` equilibrium and `42k` strongly distorted QM7-X structures for
PBE0+MBD targets, including HOMO--LUMO Gap, and separately evaluates toxicity
and lipophilicity on larger drug-like sets.

The electronic features are generated by DFTB3+MBD single points with the 3ob
parameter set. For the larger drug-like sets, the public workflow starts with
RDKit/MMFF, performs CREST/GFN2-xTB conformer search under GBSA water, keeps up
to ten low-xTB-energy conformers, computes DFTB3+MBD properties, and trains on
the lowest-DFTB-energy conformer per molecule. The supplementary table reports
the direct XGBoost SOAP-to-SOAP+D_QM Gap reduction from `2.821` to `1.442`
kcal/mol (`~0.122` to `~0.063 eV`) on equilibrium structures and from
`10.922` to `4.071` kcal/mol (`~0.474` to `~0.177 eV`) on the highly distorted
subset. These are QM7-X/PBE0+MBD results, not PCQM/B3LYP results.

### Public asset and implementation audit

The [QUED repository](https://github.com/lmedranos/QUED) exposes the descriptor
scripts, DFTB wrapper, CPU-oriented training/evaluation entry points, model
pickles, HDF5 training sets, and a replacement `ase` DFTB calculator file. The
paper states that datasets, source code, running examples, and trained models
are also archived on [Zenodo](https://doi.org/10.5281/zenodo.17106019). The
repository identifies an MIT license and pins a Python 3.9/DFTB+ 24.1-style
environment. This is a substantially stronger engineering asset than a
paper-only electronic-feature proposal.

### MolGap contract audit

QUED supports the hypothesis that a named local/global electronic teacher can
add information beyond geometry, and it supplies the right ablations: geometry
only, electronic only, and concatenated representations. It does not establish
that DFTB3+MBD frontier quantities transfer to B3LYP/6-31G* Kohn--Sham labels.
Its geometry is RDKit/MMFF plus CREST/GFN2-xTB and its target/data roles are
QM7-X/TDCommons/MoleculeNet, so using its descriptors directly would violate
the ETKDG and same-database boundaries. It is also feature concatenation, not
strict residual learning: no `target_high - target_low` label is defined.

**Disposition: A/B for public electronic-feature implementation and teacher
ablation design; C for current data, labels, weights, and geometry.** If an
electronic teacher is reopened after architecture selection, borrow the
manifest shape and the three-way ablation, but recompute one explicitly named
proxy under ETKDG and measure identity overlap, theory correlation, residual
variance, CPU cost, and blind-molecule availability before any GPU run.

## 7.19 POS-EGNN/OMol25: frontier-orbital pretraining with a physical-consistency loss

### Primary evidence

The [EES Batteries paper](https://pubs.rsc.org/en/Content/ArticleLanding/2026/EB/D6EB00024J)
uses the OMol25 DFT release to pretrain a 3D equivariant POS-EGNN on more than
`20M` electrolyte structures, with HOMO, LUMO, and Gap as the molecular targets.
Filtering for lithium-containing, non-aqueous, and 3--10-coordinate structures
leaves approximately `24k` structures for the battery-electrolyte fine-tune.
The encoder is built on GotenNet: atomic numbers and Cartesian coordinates are
passed through scalar and steerable features, attention message passing uses a
`6 Å` cutoff, and max pooling produces the graph representation.

The paper exposes a useful multi-task contract rather than only a headline
architecture. Pretraining uses AdamW, learning rate `1e-4`, batch size `16`,
and `29` epochs over the `20M` subset; fine-tuning uses learning rate
`3e-5` for `10` epochs. The four-target Huber objective assigns larger weights
to HOMO and LUMO and a smaller weight to the algebraically related Gap, while a
fourth site-charge head supplies local electronic supervision. The stated
reason is physical consistency: `Gap = LUMO - HOMO`, so the individual orbital
energies should be learned directly and the Gap should constrain, rather than
replace, them. On a held-out OMol25 electrolyte set, the small model reports
HOMO MAE `0.489 eV`, LUMO MAE `0.606 eV`, and all three target MAEs below
`0.61 eV`, with reported correlations above `0.9`.

### Public implementation audit

The [IBM/materials repository](https://github.com/ibm/materials) is Apache-2.0
and contains the `models/pos_egnn` implementation and example notebook. The
[Hugging Face model card](https://huggingface.co/ibm-research/materials.pos-egnn)
exposes Apache-2.0 `pos-egnn.v1-6M.pt` weights, inference/feature-extraction
usage, and the MPtrj provenance: `1.4M` samples used for energy/force/stress
pretraining. This is a complete code-and-weight asset for representation
inspection, but it is not the paper's OMol25 frontier-orbital checkpoint or a
PCQM-compatible target model.

### MolGap contract audit

POS-EGNN provides strong design evidence for jointly predicting HOMO/LUMO/Gap,
using HOMO/LUMO as the primary electronic quantities and Gap as a consistency
term, and separating molecular and site-level heads. It does not provide a
legal current MolGap initialization: the paper's labels are
`ωB97M-V/def2-TZVPD`, the structures are explicit MD-sampled solvent/ion-pair
configurations, and both the pretraining and fine-tuning roles are external to
the PCQM B3LYP/6-31G* and ETKDG contracts. The public MPtrj weights target
energies, forces, and stress rather than frontier orbitals.

**Disposition: B for multi-task physical-consistency and public 3D foundation
model design; C for current labels, weights, database rows, geometry, and
pretraining.** If a future HOMO/LUMO/Gap protocol is authorized, the safe first
control is a random-init same-encoder model with an explicit algebraic
consistency diagnostic, not importing POS-EGNN weights.

## 7.20 AEGCNN-MTL: related-task grouping and negative-transfer evidence

### Primary evidence

The peer-reviewed [npj Computational Materials paper](https://www.nature.com/articles/s41524-025-01917-7)
studies an adaptive edge-aware graph convolutional network with multi-head
attention and task-specific decoders. Its direct molecular evidence is QM9,
not PCQM4Mv2: the authors group the twelve QM9 properties by correlation and
physical meaning, train HOMO/LUMO/Gap together, and keep weakly related dipole,
electronic-spatial-extent, and ZPVE tasks single-task. The frontier group has a
reported Pearson correlation of `0.89` between HOMO and Gap. The reported QM9
MAEs are `23.1 meV` for HOMO, `22.0 meV` for LUMO, and `32.1 meV` for Gap;
these are not current PCQM evidence.

The same paper supplies the more important control result: on a separate
boron-doped-graphene dataset, its multi-task mode reduces band-gap MAE from
`0.0049` to `0.0031 eV` within the same architecture, but the weakly coupled
work-function task has a slightly worse RMSE under multi-task training. The
authors therefore support task selection by measured physical/statistical
relatedness rather than assuming that adding every available electronic target
must help. The paper also states that its code and BDG data are available only
from the authors on request.

### MolGap contract audit

AEGCNN-MTL is useful as a multi-task ablation and negative-transfer reference,
not as a code asset or a current candidate. QM9 uses a different data/geometry
and target-scale contract, the BDG data are not public, and no PCQM4Mv2 result or
ETKDG path is reported. The safe lesson is to measure target correlation and
negative transfer before adding HOMO/LUMO/Gap or auxiliary electronic heads;
the paper does not justify adding them to the active GraphState screen.

**Disposition: B for task-grouping and negative-transfer design; C for current
experiment, code, data, and direct PCQM claims.**

## 8. Evidence-only next actions

1. Record exact revisions/licenses for HEDMoL, Q-GEM assets, and any MET
   implementation before considering code reuse.
2. Audit canonical identity and substructure overlap for QM9/QuanDB/OP sources
   against the Track A and Track B roles.
3. If delta learning is reopened, run only the CPU proxy audit first; do not
   reserve GPU time before it passes.
4. If an electronic teacher is reopened, start with one local descriptor and
   one matched random-init GraphState control. Keep charge, bond order, OP, and
   denoising as separate protocols.
5. For the COCONUT/GFN2-xTB asset, freeze GitHub/Zenodo `v0.2.1`, record the
   MIT and upstream COCONUT terms separately, inspect the actual descriptor/data
   schema, and compare its ten-conformer xTB/Boltzmann contract with an
   ETKDG-only single-proxy audit before considering any residual experiment.
6. For QMCVNet, freeze the Caltech paper/SI hashes, verify whether its
   PubChemQC PM6/B3LYP rows overlap Track A, and preserve the PM6/MMFF-versus-
   learned comparison as a negative control. Do not reuse its geometry or
   interpret direct high-level prediction as strict delta-learning.
7. For QUED, freeze the Digital Discovery paper, MIT repository, and Zenodo
   release; inspect the D_QM field schema/model pickles and record MIT/CC-BY-3.0
   terms separately. Borrow only the geometry-only/electronic-only/combined
   ablation shape; do not calculate DFTB/CREST features on the current track or
   reserve compute until an ETKDG-compatible proxy contract is written.
8. For POS-EGNN, freeze the EES Batteries paper, IBM repository, and Hugging
   Face card separately; do not confuse the paper's OMol25 frontier checkpoint
   with the public MPtrj energy/force/stress checkpoint. If a later multi-task
   protocol is authorized, reproduce the no-teacher control and inspect
   algebraic Gap consistency before any external-weight transfer.
9. For AEGCNN-MTL, retain only the correlation-grouping and single-task versus
   multi-task control. Do not request the unpublished BDG data or treat its QM9
   errors as PCQM evidence.

## Primary-source index

- [HEDMoL paper](https://arxiv.org/html/2602.07087) and
  [code](https://github.com/ngs00/HEDMoL)
- [MET ChemRxiv record](https://chemrxiv.org/engage/chemrxiv/article-details/689e8887a94eede154d606f4)
- [Q-GEM](https://advanced.onlinelibrary.wiley.com/doi/abs/10.1002/advs.202504867)
- [OP diagram prediction](https://academic.oup.com/chemlett/article/54/3/upaf038/8058640)
- [GW/DFT delta-learning paper](https://publikationen.bibliothek.kit.edu/1000163796/151619965)
- [GW frontier-orbital dataset paper](https://www.nature.com/articles/s41597-023-02486-4),
  [arXiv record](https://arxiv.org/abs/2303.08708), and
  [Figshare archive](https://figshare.com/articles/dataset/Accurate_GW_frontier_orbital_energies_of_134_kilo_molecules_of_the_QM9_dataset_/21610077)
- [VQM24 paper](https://www.nature.com/articles/s41597-025-05428-4),
  [arXiv record](https://arxiv.org/abs/2405.05961), [Zenodo data](https://zenodo.org/records/15442257),
  [code](https://github.com/dkhan42/VQM24), and [OpenQDC adapter](https://docs.openqdc.io/stable/API/datasets/vqm24.html)
- [QCML paper](https://www.nature.com/articles/s41597-025-04720-7),
  [Zenodo metadata/examples](https://zenodo.org/records/14859804), and
  [public TFDS bucket](https://console.cloud.google.com/storage/browser/qcml-datasets/tfds/)
- [qcMol paper](https://www.nature.com/articles/s42004-026-02076-6),
  [code](https://github.com/GHUSER-haoyu/qcMol), [web server](https://structpred.life.tsinghua.edu.cn/qcmol/),
  and [parameter archive](https://zenodo.org/records/19183364)
- [QM40 paper](https://www.nature.com/articles/s41597-024-04206-y),
  [Figshare data](https://doi.org/10.6084/m9.figshare.25993060.v1), and
  [code](https://github.com/Ayeshmadu/QM40_dataset_for_ML)
- [ESA frontier-transfer paper](https://www.nature.com/articles/s41467-025-60252-z),
  [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12141427/), and
  [official code](https://github.com/davidbuterez/edge-set-attention)
- [MFGP-GEM paper](https://www.nature.com/articles/s41524-024-01479-0),
  [published PDF](https://www.nature.com/articles/s41524-024-01479-0.pdf), and
  [official code/data](https://github.com/ashah1973/MFGP-GEM/tree/main)
- [Multi-fidelity GNN transfer paper](https://www.nature.com/articles/s41467-024-45566-8),
  [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11258334/), and
  [official code](https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics)
- [Trainable dataset-embedding paper](https://doi.org/10.1088/2632-2153/ae0d41)
  and [VMDatomistic code/model](https://github.com/Fraunhofer-SCAI/VMDatomistic)
- [PubChemQC project index](https://nakatamaho.riken.jp/pubchemqc.riken.jp/),
  [86M B3LYP/6-31G*//PM6 paper](https://arxiv.org/abs/2305.18454), and
  [2017 database archive](https://nakatamaho.riken.jp/pubchemqc.riken.jp/b3lyp_2017.html)
- [MFΔML paper](https://arxiv.org/html/2410.11391),
  [MFDeltaML code](https://github.com/SM4DA/MFDeltaML), and
  [QeMFi release](https://zenodo.org/records/12734761)
- [DelFTa](https://github.com/josejimenezluna/delfta) and
  [delta-QML article](https://pubs.rsc.org/en/content/articlehtml/2022/cp/d2cp00834c)
- [GFN2-xTB/COCONUT gap paper](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00186b),
  [public workflow/data](https://github.com/sthinius87/HL-gaps-pub), and
  [Zenodo v0.2.1 archive](https://doi.org/10.5281/zenodo.15113790)
- [QMCVNet paper record](https://authors.library.caltech.edu/records/2jygg-n1r30),
  [paper PDF](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60ea947a9ab06e2e274d6cd7/original/3d-computer-vision-models-predict-dft-level-homo-lumo-gap-energies-from-force-field-optimized-geometries.pdf),
  and [ChemRxiv DOI](https://doi.org/10.33774/chemrxiv-2021-11r61)
- [QUED paper](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00411j),
  [MIT code and models](https://github.com/lmedranos/QUED), and
  [Zenodo archive](https://doi.org/10.5281/zenodo.17106019)
- [POS-EGNN/OMol25 paper](https://pubs.rsc.org/en/Content/ArticleLanding/2026/EB/D6EB00024J),
  [IBM implementation](https://github.com/ibm/materials), and
  [public MPtrj POS-EGNN weights](https://huggingface.co/ibm-research/materials.pos-egnn)
- [AEGCNN-MTL paper](https://www.nature.com/articles/s41524-025-01917-7)
