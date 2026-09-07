# Deep reading: teacher/distillation and delta-learning continuation

Date: 2026-09-07

This record extends the directly relevant reading reserve with primary-source evidence about
3D-to-2D teacher distillation, quantum-chemical delta-learning, and public implementations.
It is an evidence record, not an experiment authorization. The existing PCQM roles,
B3LYP/6-31G* targets, ETKDG train/inference contract, and database boundary remain
unchanged.

Evidence grades used here:

- A: primary paper or official documentation with enough methodological detail to reproduce
  the claim and inspect the data/target contract.
- B: public implementation or artifact with inspectable code/configuration and a real
  execution result, but incomplete scientific validation.
- C: useful engineering reference whose result is only a smoke test, synthetic test, or
  otherwise not comparable to the official task.
- Excluded: interesting method whose data, target, geometry, or evaluation contract is not
  yet auditable for MolGap.

## 1. Public 3D-prior distillation implementation: negative smoke evidence

Primary artifact: [pcqm4mv2-3d-prior-distillation](https://github.com/A-SHOJAEI/pcqm4mv2-3d-prior-distillation)

### 1.1 What the repository actually contains

The repository is explicitly organized around distilling a 3D quantum prior into a 2D
Graph Transformer for OGB-LSC PCQM4Mv2. The intended data path names the official
PCQM4Mv2 dataset and the official train/valid/test-dev/test-challenge split keys. The
configuration also exposes a teacher geometry choice:

- dft: read cached DFT coordinates;
- rdkit: construct an RDKit ETKDGv3 conformer followed by UFF optimization.

The intended model components are:

- a GIN virtual-node regression baseline;
- an EGNN teacher taking node features, edges, and positions;
- a 2D Graph Transformer student;
- optional supervised, prediction-level KD, and feature-level KD losses.

The documented loss is

    L = L_sup + alpha_kd L_kd + alpha_feature L_feat

where L_sup is student L1 regression loss, L_kd is temperature-scaled teacher/student
prediction MSE, and L_feat aligns projected student and teacher representations with MSE.
The teacher is not automatically evidence of a useful teacher: teacher quality, geometry
quality, student capacity, and the weights of the three losses are all independent
variables.

The repository's committed smoke artifact uses synthetic splits rather than the full
PCQM4Mv2 data. This distinction matters: the repository contains a full-configuration
path, but the checked-in demonstrated result is not a PCQM4Mv2 leaderboard result.

### 1.2 The executed smoke result

The README reports one run on 64 paired synthetic samples:

| model | validation MAE | throughput (mol/s) |
|---|---:|---:|
| GIN virtual-node baseline | 0.882684 | 8476.565 |
| 2D student + distillation | 1.036029 | 6491.115 |
| 2D student without distillation | 1.035819 | 7587.728 |

The reported paired bootstrap deltas are:

- baseline versus distilled student: 0.153345, CI [0.065025, 0.248129];
- baseline versus non-distilled student: 0.153135, CI [0.084303, 0.245022].

The difference between the two student variants is therefore negligible relative to
their common gap to the baseline in this smoke setup. The repository itself correctly
states that this is not evidence that distillation improves PCQM4Mv2. It records one
seed, only two teacher epochs, synthetic data, and no test-dev/test-challenge export.

### 1.3 What is reusable and what is not

Reusable engineering evidence:

- keep teacher, student, and baseline as separately switchable components;
- report a no-distillation student control;
- use paired bootstrap intervals rather than only one aggregate MAE;
- make geometry source explicit in the configuration;
- keep a split manifest and independently inspectable checkpoints.

Not yet admissible as MolGap scientific evidence:

- the positive claim that 3D distillation helps PCQM4Mv2;
- any claim based on the synthetic smoke score;
- use of the DFT geometry mode under the current ETKDG inference contract;
- substitution of ETKDGv3+UFF for the repository's exact MolGap conformer protocol without
  a contract review.

Disposition: B for implementation structure, C for scientific performance evidence, and
negative evidence against assuming that adding KD terms alone will improve a smaller
2D student. This is a useful failure-control reference, not a promoted candidate.

## 2. D&D: denoising and distillation from DFT conformers

Primary source: [D&D: Distilling the Knowledge in 3D Molecular Structures for
 2D Graph Transformers](https://ojs.aaai.org/index.php/AAAI/article/download/31986/34141)

### 2.1 Pretraining contract

D&D creates a 3D teacher from a perturbed stabilized conformer. For coordinates R_i, the
pretraining input is perturbed as

    R_tilde_i = R_i + sigma epsilon_i

and the 3D encoder predicts the injected noise with a denoising L2 objective. The paper
uses a TorchMD-NET SE(3)-equivariant encoder for this 3D stream.

The 2D and 3D representations are then aligned by cross-modal distillation. The paper
describes both:

- graph-level alignment between mean-pooled 2D and 3D representations;
- node-level alignment between corresponding atom representations.

The 3D stream is frozen during the relevant distillation stage, so gradients update the
2D student. For downstream work, the 3D teacher is discarded and the 2D graph model is
fine-tuned, with L1 regression or binary cross-entropy classification as appropriate.

### 2.2 Data and leakage boundary

The paper reports pretraining on PCQM4Mv2's approximately 3.7 million molecules. Each
molecule is paired with a single lowest-energy DFT conformer. The paper states that the
Gap labels are not used in pretraining: the graph/conformer pairs are used as
unlabeled geometry data for denoising and distillation.

This is an important distinction for MolGap:

- it avoids direct target-label leakage in the pretraining loss;
- it still uses a privileged coordinate source;
- the coordinate source is a DFT conformer, not the current ETKDG inference contract.

The downstream study covers MoleculeNet and curated regression tasks. The node-level
distillation variant is motivated by the fact that atom-corresponding supervision can
preserve more topology-sensitive information than a single pooled molecular vector.

### 2.3 What can be learned without importing its contract

Strong evidence:

- a teacher can transfer geometry-sensitive information without being present at
  inference;
- node-level correspondence is a reasonable alternative to only aligning a pooled
  molecular vector;
- the no-label pretraining boundary is conceptually compatible with a fixed-label
  supervised dataset.

Contract conflict:

- D&D's single lowest-energy DFT conformer is privileged information relative to MolGap's
  ETKDG-only deployment;
- replacing it with DFT geometry would create a geometry mismatch even if the student is
  2D at inference;
- simply calling the result a 2D model would not remove the privileged-teacher issue.

Disposition: A for the training design and data-use statement, but not a directly
deployable MolGap protocol. A future admissible variant would have to use the same
ETKDG-derived conformer source for teacher pretraining and inference, or be explicitly
registered as a privileged-teacher study with a separate scientific contract.

## 2A. Coordinating Cross-modal Distillation (CCMD): global/local teacher loss with size normalization

Primary source: [Coordinating Cross-modal Distillation for Molecular Property
Prediction](https://arxiv.org/html/2211.16712). The paper's abstract says that
code would be released, but no author-owned executable repository or checkpoint
was verified in the current audit. This is therefore paper evidence, not a
ready-to-run asset.

### Method

CCMD trains a 3D Graphormer teacher, freezes it, and distills into a 2D
Graphormer student. The two streams share the same 12-layer, width-768
Graphormer backbone family but use different input relations:

- the 2D stream uses adjacent chemical-bond features;
- the 3D stream uses RBF-expanded interatomic distances;
- an absolute position encoding is formed by aggregating adjacent edge
  information into the initial atom token.

The distillation loss is explicitly split into two roles. The virtual token is
the global molecular representation, so CCMD aligns the virtual token at every
Transformer layer. The atom tokens are aligned separately as local
distillation. The paper argues that summing local losses over a molecule with
`N` atoms makes gradients grow with molecular size, and derives a coordinating
factor of order `1/N^2` for a Transformer (`1/N` for a simple GNN). In the
reported Graphormer implementation the local atom loss is first converted to a
mean, and the coordinating experiment uses an additional `1/N` factor; this
distinction must be preserved rather than summarized as one universal formula.

This is a useful teacher-design detail: pooled/global alignment and atom-level
alignment are not interchangeable, and a local loss can hurt when its size
dependence is left uncontrolled.

### Direct PCQM evidence and controls

The paper adds DFT coordinates to PCQM4Mv2 training and distillation but
evaluates the validation molecules with 2D inputs. It uses batch `512`, 70
epochs, Adam with initial learning rate `2e-4` and momentum `0.9`, on eight
V100 GPUs. Its main validation table reports:

| model or control | validation MAE (eV) |
|---|---:|
| Graphormer | 0.0864 |
| reproduced baseline + APE | 0.0845 |
| global distillation, all layers | 0.0822 |
| local atom loss only (mean) | 0.0870 |
| global + manually searched local weight | 0.0818 |
| global + size-coordinated local weight | 0.0809 |

The paper's abstract additionally states a `0.0734` test-challenge MAE and
fourth place in the 2022 OGB-LSC challenge, while the accessible main table is
a validation comparison. The two numbers must not be merged into one
comparable score. The table also contains a `baseline+3D` value of `0.040`,
which is a 3D-input control and not the deployable 2D student; it is not a
valid comparison for the current inference contract.

The ablation direction is more reliable than the headline: global molecular
distillation produces a small gain, naive local atom distillation degrades the
baseline, all-layer distillation is better than last-layer-only alignment, and
size coordination restores part of the local signal. On MolHIV, where the
authors generate imperfect RDKit conformers, global distillation still helps
but the paper explicitly acknowledges that the weak teacher limits the result.

### MolGap contract audit

CCMD is one of the strongest direct teacher papers for the target family, but
it is not a current MolGap result for four reasons:

1. DFT coordinates are used for teacher training/distillation, while MolGap
   requires ETKDG at both training and inference whenever geometry is present;
2. the validation split follows an earlier Graphormer protocol rather than a
   newly verified current official role manifest;
3. the 0.0809 table value is a 68M-ish Graphormer-scale comparison, not the
   bounded GraphState budget;
4. no reproducible code/checkpoint was found, so exact row alignment, teacher
   stopping, and feature-detachment details remain unverified.

The safe lesson is a future ETKDG-only teacher protocol with three paired
controls: 2D student without teacher, global-token-only alignment, and global
plus size-normalized node alignment. Teacher outputs must be frozen artifacts
with row IDs, geometry hashes, and a declared train-only role. A teacher trained
on DFT PCQM coordinates cannot be called ETKDG-compatible merely because the
student runs in 2D at inference.

**Disposition: A for global/local distillation and negative-transfer evidence;
B/C for current MolGap use.** CCMD strengthens the case for a separately
authorized ETKDG teacher study, but it does not authorize a run, a checkpoint,
or a database change.

## 3. DelFTa: quantum-chemical delta-learning with an explicit baseline

Primary sources:

- [DelFTa article landing page](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d2cp00834c)
- [DelFTa public implementation](https://github.com/josejimenezluna/delfta)

### 3.1 Core idea

DelFTa learns the correction from a cheaper reference calculation to a higher-level
quantum-chemical target rather than learning the target directly. Its models are
E(3)-invariant 3D message-passing networks for molecular, atomic, and bond-level
properties.

The published comparison is on QMugs, approximately two million molecular conformers.
The reference is GFN2-xTB and the target is a higher-level DFT calculation
(ωB97X-D/def2-SVP in the reported benchmark). The model therefore requires the
reference calculation or its output at inference.

Representative reported errors include:

| property | direct | delta |
|---|---:|---:|
| HOMO | 35.0 meV | 36.7 meV |
| LUMO | 36.8 meV | 27.8 meV |
| HOMO-LUMO gap | 52.9 meV | 47.3 meV |
| dipole moment | 0.1588 D | 0.0946 D |
| Mulliken charges | 0.0029 | 0.0027 |
| Wiberg bond order | 0.0017 | 0.0011 |

The same article reports very large errors for the raw GFN2-xTB reference on some
properties, including approximately 2115 meV for HOMO, 7773 meV for LUMO, and 5658 meV
for the gap. The practical lesson is not that any cheap baseline works: delta-learning
depends on a baseline that is sufficiently correlated with the target while remaining
cheap enough to obtain at inference.

The paper also reports a direct-vs-delta exception for HOMO, where direct learning is
slightly better. Delta is therefore a hypothesis about bias structure and data
efficiency, not a universal replacement for direct regression.

### 3.2 Why it is not directly transferable

The following contracts differ from MolGap:

- QMugs rather than the retained PCQM4Mv2/Track A database;
- GFN2-xTB to ωB97X-D/def2-SVP rather than B3LYP/6-31G* Kohn-Sham outputs;
- explicit 3D conformers and an E(3)-invariant model;
- a required cheap reference calculation at inference.

The result is still strong evidence for a future same-database delta study, provided that
the cheap reference is computed on the same retained rows, its theory and units are
frozen, and the inference cost is accepted. It does not justify changing the database or
silently replacing B3LYP targets.

Disposition: A for the delta principle and baseline-correlation evidence; excluded from
the current MolGap candidate list until a same-database baseline and cost protocol is
written and separately authorized.

## 4. Selected ML versus delta-QML for HOMO-LUMO gaps

Primary source: [Selected machine learning of HOMO-LUMO gaps with improved
 data-efficiency](https://pubs.rsc.org/en/content/articlehtml/2022/ma/d2ma00742h)

### 4.1 Experimental design

This study compares direct and delta-style quantum machine learning in a small-data
setting. It uses QM7b with a ZINDO reference and GW-level targets, together with QM9
B3LYP data for a related analysis. Rather than treating all molecules as one homogeneous
sample, the study selects chemically simple classes such as aromatic-plus-carbonyl,
unsaturated, and saturated molecules before constructing learning curves.

The paper reports that selected ML can reach roughly 0.1 eV error with up to an
order-of-magnitude fewer molecules in the examined setup. It also reports that the
selected direct model can be more data-efficient than the delta model in that particular
comparison.

### 4.2 Evidence-level lesson

This is a useful counterexample to the slogan “delta-learning is always more
data-efficient.” A delta model can benefit when the baseline captures a large,
structured part of the target error. A selected direct model can win when chemical
routing makes the target easier before the correction task is formed.

The design lesson that transfers without changing MolGap's database is not to create a
new dataset. It is to test, in a separately registered protocol:

- whether a cheap same-row baseline correlates with each target;
- whether residual variance is lower than target variance;
- whether any chemical stratification is used only as a declared split/training
  analysis rather than a hidden split change;
- whether direct and delta models are paired on exactly the same rows and seed.

The study does not authorize importing its QM7b, QM9, ZINDO, or GW labels into MolGap.

Disposition: A for the direct-versus-delta comparison and the data-efficiency warning;
not a directly comparable PCQM score.

## 5. Δ-DFT and GW corrections: related theory, different target

Two additional primary lines of evidence clarify what “delta” may mean.

### 5.1 Δ-DFT

[Quantum chemical accuracy from density functional approximations via machine
 learning](https://www.nature.com/articles/s41467-020-19093-1) learns a correction from
a lower-level DFT description toward a coupled-cluster-quality result. In the reported
water and ethanol tests, the correction can reach useful accuracy with fewer samples than
a direct high-level model, but the correction is tied to the specific low-level/high-level
pair.

This is a theoretical justification for residual learning, not a HOMO/LUMO benchmark.
Its key warning transfers directly: a delta model is not portable across target theories
unless the theory pair is part of the model contract.

### 5.2 DFT-to-GW quasiparticle correction

[Interpretable delta-learning of GW quasiparticle energies from GGA-DFT](https://doi.org/10.1088/2632-2153/acf545)
uses a lower-level DFT description to predict a higher-level GW correction for
HOMO/LUMO/GAP-like quantities. This is closer to the MolGap target family than
energy-only Δ-DFT, but it still changes the target theory and typically uses orbital or
atom-wise information that is unavailable in the current B3LYP/6-31G* label contract.

Disposition: both are strong conceptual references for residual theory correction, not
admissible replacement targets for MolGap.

## 6. A safe same-database delta protocol, if later authorized

The following is a gate specification, not an authorization to run it.

### Gate 0: identity and inference cost

Before any model is trained, record:

- exact target theory and units;
- exact cheap-proxy theory, software, charge, spin, geometry input, and failure policy;
- whether the proxy is available for every inference molecule;
- total CPU cost and storage cost;
- whether the proxy uses the same ETKDG conformer or an additional geometry optimization.

A proxy that cannot be produced for the deployment population is not a valid inference
feature merely because it improves an offline score.

### Gate 1: train-only residual audit

On the official training role only, compute or retrieve the proxy for a fixed audit subset
and publish:

- correlation between proxy and each target;
- target standard deviation;
- residual standard deviation;
- residual mean and heavy-tail statistics;
- missing/failure rate;
- leakage check confirming that no validation/test target was used to choose the proxy.

Do not spend GPU time on delta-learning unless the residual is materially more predictable
than the original target or a clear physical reason justifies the extra cost.

### Gate 2: paired direct control

For one seed-42 screen, train:

1. the existing direct control;
2. an identical-capacity residual model;
3. optionally, a residual-plus-direct head if the residual is biased.

Use exactly the same rows, graph cache, ETKDG conformers, optimizer schedule, and
evaluation code. The delta prediction must be reconstructed as

    target_hat = proxy + residual_hat

in the target's declared units. A direct model must remain the paired control.

If the proxy itself is learned, its training must be out-of-fold for the residual audit;
otherwise the residual can be artificially compressed by in-sample proxy fitting.

### Gate 3: deployment and promotion

A delta candidate is not promotable unless:

- its proxy is reproducible on deployment inputs;
- direct and delta scores are paired on the same official role;
- the gain survives the declared seed and uncertainty policy;
- the proxy theory and geometry are recorded;
- the checkpoint stores the proxy metadata needed to reconstruct predictions.

This protocol preserves the current database and prevents a delta label or privileged
geometry source from entering by implication.

## 7. QeMFi: a reproducible multi-fidelity benchmark for delta-learning cost claims

### Primary evidence

- Paper: [QeMFi: A Multifidelity Dataset of Quantum Chemical Properties of Diverse Molecules](https://www.nature.com/articles/s41597-024-04247-3), with the [open arXiv record](https://arxiv.org/abs/2406.14149).
- Data: [Zenodo record 13925688](https://doi.org/10.5281/zenodo.13925688).
- Code: [official QeMFi repository](https://github.com/vivinvinod/QeMFi).

QeMFi is valuable less for its chemical coverage than for its explicit
multi-fidelity accounting. It samples 15,000 geometries for each of nine
molecules from the WS22 excited-state/configuration-space collection, giving
135,000 geometries. For each geometry, ORCA 5.0.1 computes five TD-DFT
fidelities using CAM-B3LYP and increasing basis sets:
STO-3G, 3-21G, 6-31G, def2-SVP, and def2-TZVP. The records include ground-state
SCF energies, ten vertical excitation energies, oscillator strengths, transition
dipoles, molecular dipoles, rotational constants, and per-fidelity single-core
calculation times.

The repository contains the ORCA input/extraction scripts, data-generation
scripts, learning-curve tooling, and MFML/o-MFML kernel-regression benchmarks.
The paper validates the time-benefit idea on ground-state energies and
excitation energies: adding cheaper fidelities shifts the learning curves to
lower data-generation cost, with a reported cumulative ground-energy example
showing about a six-fold time benefit for an STO-3G baseline. This is a
benchmark result for QeMFi's nine molecules, not a PCQM Gap result.

### What transfers to MolGap

The strongest transferable lesson is procedural: a delta or multi-fidelity
claim must report (1) the exact lower and higher target theories, (2) paired
identities and geometries, (3) the cost of each label, (4) a direct
single-fidelity control, and (5) learning curves against label-generation cost.
The paper also distinguishes MFML from an optimized variant whose validation
set has a target-fidelity cost; that cost must not be silently omitted.

QeMFi does not authorize a current experiment. Its nine-molecule scope,
TD-DFT/excitation target, Wigner/geodesic sampling, and non-ETKDG geometries
are incompatible with the current B3LYP/6-31G* Gap contract. It can be used
only as a protocol reference if a same-database PCQM delta study is explicitly
opened later.

Disposition: **A for multi-fidelity protocol and cost-accounting evidence; C
for current-target scientific performance**. No database change or experiment
is authorized.

## 8. ViSNetGWBSE: target-aligned low-fidelity pretraining for qsGW and GW-BSE

### Primary evidence

- Paper: [Transfer learning of GW Bethe--Salpeter equation excitation energies](https://pubs.rsc.org/en/content/articlehtml/2026/sc/d5sc09780k), Chemical Science 17, 8090--8099 (2026); the [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12951312/) and [arXiv record](https://arxiv.org/abs/2512.11596) are also available.
- Code, checkpoints, and test data: [official ViSNetGWBSE repository](https://github.com/daoiradrio/ViSNetGWBSE).
- High-fidelity data: [QM9GWBSE Zenodo record](https://zenodo.org/records/17902233), whose public release lists `edft`, `eexc_ss`, `eexc_st`, `eqp`, oscillator-strength, transition-dipole, and XYZ archives.
- Low-fidelity sources named by the paper: [OMol25](https://huggingface.co/facebook/OMol25/blob/main/DATASET.md) and [QCDGE](https://langroup.site/QCDGE/).

### 8.1 What was actually compared

This is a clean multi-fidelity transfer study rather than a generic claim that
"more pretraining helps." The model is ViSNet, an SE(3)-equivariant message-passing
network that uses distances, angles, dihedrals, and improper dihedrals. Two sizes
are reported: approximately 0.89M and 2.5M trainable parameters. The public
configuration makes the small implementation concrete: `lmax=2`, 216 hidden
channels, 25 Gaussian radial basis functions, 5.0 A cutoff, three message-passing
layers, batch size 16, 50 epochs, and an AdamW warmup/cosine schedule. The code
sets `torch.manual_seed(42)` and supports both full fine-tuning and a frozen
representation with a trainable readout.

The source/target contracts are explicit:

- DFT pretraining uses up to 10M neutral OMol25 molecules with HOMO, LUMO, and
  Gap at omegaB97M-V/def2-TZVPD; 1M and 5M subsets are also compared.
- TDDFT pretraining uses QCDGE excitation energies at omegaB97X-D/6-31G(d).
- Fine-tuning uses qsGW quasiparticle energies or qsGW-BSE excitation energies
  from QM9GWBSE. The public default configuration has 120,000 training rows,
  10,000 validation rows, and 3,885 test rows.
- The paper evaluates four held-out regimes: PC9-like chemistry, larger
  molecules with the QM9 element set, and a heteroatom-shifted set. It applies
  SMILES-based deduplication across pretraining, fine-tuning, and test data.

The authors compare two transfer contracts: **Full**, which updates the complete
network, and **Transfer**, which updates only the readout after message passing.
This is important evidence because it separates representation reuse from merely
having a favorable initialization.

### 8.2 Results that survive the evidence audit

Across the reported qsGW and qsGW-BSE tests, pretraining lowers error and is
especially useful for reducing large outliers. The authors use independent random
subsets of 10k, 20k, 40k, 80k, and 120k high-fidelity rows, with three samples at
each size. On the most challenging test sets, pretrained models reach the
full-data error regime with about 20k rather than 120k qsGW fine-tuning rows, and
about 40k rather than 120k qsGW-BSE rows. These are convergence/data-demand
claims for the paper's qsGW targets, not claims about PCQM4Mv2 Gap MAE.

The paper selects 5M low-fidelity pretraining examples as a practical compromise
for the small transfer model. This is a useful cost result: the method does not
require using the largest available low-fidelity corpus to obtain a measurable
benefit. It also shows that target alignment matters: TDDFT excitation pretraining
is more useful for neutral excitation targets than DFT MO-energy pretraining in
some of the GW-BSE comparisons, while the broader DFT pretraining can be helpful
for quasiparticle energies.

### 8.3 What can transfer to MolGap, and what cannot

The strongest transferable principle is **theory- and target-aligned transfer**:
pretraining should approximate the physical quantity that the high-fidelity head
must predict, and the study should compare full fine-tuning with readout-only
fine-tuning. It also supplies a reproducible learning-curve template: vary the
amount of target-role data, use independent samples, keep a no-pretraining
control, and report outliers rather than only the mean MAE.

It is not a direct MolGap experiment. OMol25 and QCDGE are external databases;
their functionals/bases differ from B3LYP/6-31G*, and qsGW/qsGW-BSE targets are
not B3LYP Kohn--Sham HOMO/LUMO/Gap. The paper also uses explicit Cartesian
coordinates, whereas MolGap's active contract is ETKDG at both training and
inference. Importing a checkpoint or external rows would therefore change the
database and/or geometry contract. No such import is permitted by this record.

If this line is ever opened, the minimum candidate gate is: freeze the source and
target theory pair; verify exact molecular identity and role overlap; preserve
ETKDG or explicitly authorize a new geometry contract; compare direct training,
full transfer, and readout-only transfer on identical rows; and report target-data
cost and outlier behavior. The paper is strong evidence for that protocol, not an
authorization to run it now.

Disposition: **A for the multi-fidelity transfer principle and reproducible
full-vs-readout design; B/C for current MolGap use because the database, theory,
and geometry contracts do not match.**

## 9. Candidate ledger after this continuation

| candidate | evidence | current disposition |
|---|---|---|
| public PCQM 3D-prior-distillation repo | B/C | engineering reference and negative smoke evidence; no positive claim |
| D&D node-level distillation | A | strong teacher template; DFT conformer contract blocks direct adoption |
| CCMD global/local distillation | A/B/C | direct PCQM validation evidence and size-normalized negative-transfer ablations; DFT geometry, non-current split audit, and no verified code block current use |
| DelFTa | A | strongest delta reference; requires same-database proxy and theory/cost gate |
| selected HOMO-LUMO delta-QML | A | direct-vs-delta caution and chemical-stratification reference |
| Δ-DFT | A | theory-only residual-learning support; not a current target |
| DFT-to-GW delta | A | related orbital correction; target-theory mismatch |
| QeMFi | A/C | reproducible multi-fidelity cost protocol; no current Gap result |
| ViSNetGWBSE | A/B/C | strongest recent full-vs-readout multi-fidelity transfer reference; external qsGW/GW-BSE contract |

No new database, public label, DFT conformer, or experiment has been added to MolGap by
this record.

Primary sources for this continuation are the [QeMFi paper](https://www.nature.com/articles/s41597-024-04247-3),
[Zenodo data](https://doi.org/10.5281/zenodo.13925688), and
[official code](https://github.com/vivinvinod/QeMFi).

Additional primary sources for this continuation are the [CCMD paper](https://arxiv.org/html/2211.16712), the [ViSNetGWBSE paper](https://pubs.rsc.org/en/content/articlehtml/2026/sc/d5sc09780k),
[official code](https://github.com/daoiradrio/ViSNetGWBSE), and
[QM9GWBSE target-data record](https://zenodo.org/records/17902233).
