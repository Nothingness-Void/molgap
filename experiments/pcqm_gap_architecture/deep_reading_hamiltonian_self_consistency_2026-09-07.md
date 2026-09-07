# Deep Reading: Hamiltonian, Self-Consistency, and Electronic Pretraining (2026-09-07)

This record audits methods that learn a Kohn--Sham Hamiltonian, density map,
SCF update, or electronic operator rather than only a scalar molecular
property. It also audits public Hamiltonian datasets, code, and checkpoints
that can inform a pretraining, teacher, or delta-learning protocol.

The audit is evidence-first. A method is not a current MolGap experiment merely
because it reports a lower Hamiltonian error or a strong result on QH9. The
active contract remains the existing PCQM4Mv2 / repaired-2M PubChemQC database,
B3LYP/6-31G* Kohn--Sham HOMO/LUMO/Gap labels, ETKDG for both training and
inference, and the repository's frozen random-init architecture gate.

## Executive conclusion

The strongest new evidence is a coherent progression:

1. **QH9** makes Hamiltonian prediction reproducible with public data, splits,
   baselines, and checkpoints.
2. **Self-Consistency Training**, **DEQHNet**, and **NeuralSCF** turn the
   electronic calculation into a differentiable or fixed-point objective,
   rather than treating a matrix or density as an arbitrary regression target.
3. **HamEvo** is the clearest recent operator-level Delta-learning design: it
   predicts an SCF-like local correction `Delta H`, iterates it to a fixed point,
   and transfers across size and exchange-correlation functional.
4. **HELM** provides the clearest controlled pretraining evidence: Hamiltonian
   pretraining, followed by a frozen or fine-tuned energy head, improves
   low-data energy prediction. It does not provide a same-contract frontier
   result, and the paper still marks code and data as forthcoming.
5. **QHFlow2** is the most useful public checkpoint asset in this batch for
   frontier analysis: its paper reports HOMO, LUMO, and Gap from a predicted
   Hamiltonian, while its official repository and Hugging Face cards expose
   runnable checkpoints. Its B3LYP/def2-SVP Hamiltonians, explicit 3D input,
   and QH9 splits keep it outside the current PCQM/ETKDG experiment.

No source in this batch clears the exact current admission gate. No database,
checkpoint, pretrained initialization, external geometry, or remote experiment
is imported or authorized by this record.

## 1. QH9: public Hamiltonian benchmark and asset surface

### Primary evidence

- [QH9 paper](https://arxiv.org/html/2306.09549)
- [official AIRS/QHBench implementation](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9)
- [QH9 Zenodo record](https://zenodo.org/records/8274793)

### What is actually released

QH9 contains 130,831 stable QM9-derived molecular geometries and dynamic
trajectories for either 999 or 2,998 molecules. The released tasks include
stable in-distribution, stable size-OOD, dynamic geometry-OOD, and dynamic
molecule-OOD splits. The expanded 300k dynamic task contains 299,800
geometries.

The reference calculations use PySCF with B3LYP/Def2SVP, a tight SCF
threshold, and a grid level of 3. The stored objects include the Hamiltonian,
overlap, coefficient/eigenvalue-related quantities, atom numbers, and 3D
positions. The benchmark evaluates Hamiltonian MAE, orbital-energy error,
wavefunction coefficient similarity, and whether the prediction improves SCF
initialization.

The official repository supplies dataset loaders, generation examples,
baseline QHNet code, pretrained-model loading, and scripts for the DFT
acceleration check. This is a genuinely reproducible benchmark asset, not just
a paper table.

### MolGap decision

QH9 is a benchmark and design reference only. It is not merged into PCQM or
Track A. Its explicit Hamiltonian/overlap labels and 3D geometry contract are
not present in the current scalar PCQM database, and importing its rows would
change both the target theory and the leakage surface.

## 2. Self-Consistency Training for DFT Hamiltonian Prediction

### Primary evidence

- [ICML 2024 paper](https://proceedings.mlr.press/v235/zhang24ak.html)
- [arXiv full text](https://arxiv.org/html/2403.09560)
- [official QHNet code base used by the paper](https://github.com/divelab/AIRS)

### Method and results

The model predicts a mean-field DFT Hamiltonian from molecular structure. A
Hamiltonian is then diagonalized with the overlap matrix, occupations are
assigned, an electron density is reconstructed, and a new Hamiltonian is
reconstructed from that density. The self-consistency loss compares the
predicted Hamiltonian with the reconstructed one, so the unlabeled objective
does not require a reference Hamiltonian for every structure.

The paper explicitly requires differentiating through the eigensolver and
Hamiltonian reconstruction. Detaching the eigenvectors gives the wrong
gradient and can degrade the model. Near-degenerate orbital gaps cause unstable
gradients; the implementation controls this with a truncated inverse-gap factor
and skips updates with excessive gradients. Density fitting and GPU grid
evaluation reduce the reconstruction cost from a naive fourth-order scaling to
approximately cubic scaling.

The experiments use MD17 with PBE/def2-SVP and QH9 with B3LYP/def2-SVP. The
data-scarce setting uses 100 labeled MD17 structures per molecule and unlabeled
structures for self-consistency. On QH9 OOD and MD22 transfer settings,
self-consistency fine-tuning improves derived orbital quantities, including
LUMO and HOMO--LUMO Gap, even when fully supervised Hamiltonian MAE is not
always the best metric.

### MolGap decision

This is a high-value teacher/pretraining objective reference, but it needs
Hamiltonian, overlap, basis-integral, density, and grid operators absent from
the current database. The paper does not provide a dedicated PCQM/ETKDG
frontier implementation. It is not a current experiment.

## 3. DEQHNet: fixed-point Hamiltonian prediction

### Primary evidence

- [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/a31d1942f4f4f52a73bfd1d67856bed7-Paper-Conference.pdf)
- [official DEQHNet repository](https://github.com/Zun-Wang/DEQHNet)

### Method and results

DEQHNet treats Hamiltonian prediction itself as a deep-equilibrium fixed-point
problem. The model consumes geometry, overlap, and a Hamiltonian state and
solves for a fixed point instead of stacking a large finite-depth sequence of
updates. The training objective avoids DFT calculations during the learned
fixed-point training loop.

The paper evaluates MD17 PBE/def2-SVP and QH9 B3LYP/def2-SVP. It reports a
useful warning: lower Hamiltonian MAE does not automatically imply lower
orbital-energy MAE. The QH9 tables show that coefficient similarity,
Hamiltonian error, and derived orbital errors can move in different directions;
the overlap input is also important in ablations.

The repository is public and includes model, data, configuration, and training
instructions. Its dependencies include PyTorch, PyG, PySCF, and Psi4, so it is
an external operator-learning package rather than a drop-in scalar Gap model.

### MolGap decision

Use the fixed-point and overlap-ablation ideas as mechanism references. Do not
interpret its QH9 matrix metrics as direct PCQM Gap evidence. No current
experiment is admitted because the active database has no compatible H/S
labels or SCF operator surface.

## 4. QHFlow: flow matching plus frontier-aware readout

### Primary evidence

- [NeurIPS 2025 paper](https://arxiv.org/html/2505.18817)
- [official QHFlow repository](https://github.com/seongsukim-ml/QHFlow)

### Method and results

QHFlow models the Hamiltonian as a structured distribution and learns an
SE(3)-equivariant flow-matching vector field that transports a simple matrix
prior to the target Hamiltonian. A post-hoc weighted alignment fine-tuning
stage explicitly emphasizes orbital-energy errors, including HOMO, LUMO, and
Gap derived from the predicted matrix.

The paper evaluates four MD17 molecules with PBE/def2-SVP and QH9 with
B3LYP/def2-SVP. The QH9 benchmark reports frontier quantities and uses stable
ID/OOD plus dynamic geometry/molecule splits. The paper also explains that
PubChemQH was not used because the data and code were not public at the time.

The official repository exposes dataset loaders, QH9/MD17 configurations,
LMDB sharding, DFT utilities, training and inference entry points, and a
checkpoint directory. The repository README says some pretrained checkpoints
are available upon request and that fine-tuning is not yet fully working, so
the code surface is stronger than the artifact surface.

### MolGap decision

The frontier-weighted readout is a valuable design reference, but flow
matching over basis-dependent Hamiltonians is not a scalar Gap method. The
current database does not supply the required matrices, and the geometry/theory
contract is external. No experiment is admitted.

## 5. nablaDFT: the original public conformational Hamiltonian dataset

### Primary evidence

- [PCCP paper](https://doi.org/10.1039/D2CP03966D)
- [official repository and data links](https://github.com/AIRI-Institute/nablaDFT)

### What it contributes

nablaDFT established a public large-scale molecular conformational benchmark
with energy and Hamiltonian prediction tasks, database files, data loaders,
model registries, and checkpoint links. Its design is important because it
keeps molecule/conformation identity and Hamiltonian/overlap matrices as first-
class records instead of reducing everything to a scalar table.

The theory is ωB97X-D/def2-SVP with Psi4 and the data are generated from
drug-like conformational structures. The repository exposes database access,
wavefunction files, summary tables, and train/test splits.

### MolGap decision

This is a public data-contract and loader reference only. The target functional,
basis, conformer generation, and explicit Hamiltonian labels differ from
B3LYP/6-31G*/ETKDG. No rows or wavefunctions are imported.

## 6. ∇²DFT: public Hamiltonian, frontier, force, and wavefunction benchmark

### Primary evidence

- [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/40d45b1e23d00d5895e65778e85cf8ee-Paper-Datasets_and_Benchmarks_Track.pdf)
- [official nablaDFT repository](https://github.com/AIRI-Institute/nablaDFT)
- [version-2 arXiv record](https://arxiv.org/abs/2406.14347)

### What is actually available

The expanded dataset contains 1,936,929 molecules and 12,676,264 conformations
over H, C, N, O, F, S, Cl, and Br. Each conformation can include energy,
forces, Hamiltonian and overlap matrices, orbital coefficients, and a Psi4
wavefunction object. The metadata explicitly includes DFT HOMO, DFT LUMO, and
DFT HOMO--LUMO Gap fields.

The data are generated from MOSES using RDKit conformers, Butina clustering,
and a coverage rule that retains between 1 and 69 conformers in the paper
version. The labels use ωB97X-D/def2-SVP in Psi4. The benchmark supplies
conformation, scaffold, and structure splits, an extendable model framework,
and checkpoint links.

### MolGap decision

This is the strongest public external asset in this batch for studying how
operator labels, frontier metadata, wavefunctions, and conformer identity can
coexist in one database. It is not an import candidate: the basis/functionals,
conformer pipeline, and label contract are different, and the repository's
large archives would violate the current database-preservation rule.

## 7. NeuralSCF: density-map pretraining plus implicit fixed-point fine-tuning

### Primary evidence

- [npj Computational Materials paper](https://doi.org/10.1038/s41524-026-02110-0)
- [arXiv full text](https://arxiv.org/html/2406.15873)
- [official code](https://github.com/songfeitong/neuralscf)
- [official data record](https://doi.org/10.57760/sciencedb.37834)

### Method and results

NeuralSCF represents the density with atom-centred Gaussian auxiliary-basis
coefficients. The network maps geometry plus the current density to the next
density, and Pulay mixing iterates the map to a fixed point. A final KS step
derives orbital quantities and energy from the converged density; the network
is not a direct scalar HOMO/LUMO/Gap head.

Training has two stages. Explicit pretraining uses every available SCF
trajectory pair with an L2 density loss. Implicit fine-tuning then minimizes a
fixed-point density loss through the implicit-function theorem. On QM9 the
reported fine-tuned density NMAE is 0.064%, energy MAE is 0.010 kcal/mol, and
HOMO--LUMO Gap MAE is 7.3 meV. The paper also reports MD17 ethanol and MD22
ALA3 tests plus zero-shot perturbation and bond-rotation checks.

### MolGap decision

This is strong evidence for explicit-trajectory pretraining followed by
implicit self-consistency fine-tuning, and the code/data surface is public.
The current PCQM database has neither density trajectories nor the auxiliary
basis/KS solver needed for the loss. No external density data or checkpoint is
used.

## 8. HamEvo: fixed-point residual Hamiltonian Delta-learning

### Primary evidence

- [HamEvo arXiv record](https://arxiv.org/abs/2606.14498)
- [full paper](https://arxiv.org/html/2606.14498)
- [official code](https://github.com/axdfhj/HamEvo_official)
- [official Hugging Face data](https://huggingface.co/datasets/ZJUSCL/hamevo-data)

### Method and results

HamEvo learns an SCF-like operator
`H^(t+1) = F_theta(H^(t); G)` and predicts a local correction `Delta H` rather
than only an absolute Hamiltonian. Anderson iteration is used in the forward
fixed-point solve and implicit differentiation uses a Broyden-style backward
solve. The architecture uses atom-pair diagonal/off-diagonal blocks and an
equivariant transformer with maximum degree four.

The training protocol is explicit: SCF-transition trajectory pretraining,
fixed-point Hamiltonian and density-matrix supervision, then size/functional
transfer. The default reference is B3LYP/def2-SVP with a 1e-10 Ha SCF
tolerance. GDB17, xTB-MD geometries, QMugs, nablaDFT-derived external data,
and recomputed MD17/QH9/evolution/equilibrium/target sets are exposed in the
paper/data package.

On the QMugs test set the paper reports HOMO MAE 0.036 eV and LUMO MAE 0.053
eV. It also reports twenty-shot transfer to much larger molecules and
cross-functional adaptation to ωB97X-D, PBE0, SCAN0, ωB97X, and ωB97M-V. The
paper notes fixed def2-SVP basis and no force prediction as limitations.

### MolGap decision

HamEvo is the clearest recent operator-level Delta-learning reference, but
`Delta H` is not the same as a scalar `Delta Gap`. Its SCF trajectories,
Hamiltonian/density labels, explicit coordinates, and basis/functionals are
external to the active database. The public code/data are recorded as a
reference, not imported or authorized for current use.

## 9. Self-consistent Validation for Machine Learning Electronic Structure

### Primary evidence

- [ICML paper/preprint](https://arxiv.org/html/2402.10186)
- [accepted manuscript record](https://ora.ox.ac.uk/objects/uuid:0d7d2d34-c3ed-4c29-a9e9-49d8f249a6d4)

### Method and results

This work uses the physics residual
`e = H D S - S D H` as a label-free self-DIIS confidence signal. It predicts
Hamiltonian and density matrices, measures the residual with the model's own
output, and uses the residual for validation and active-learning decisions. On
RMD17 and QM9, the residual correlates with strict DIIS residual, MAE, total
energy, and HOMO--LUMO Gap error, including perturbed OOD structures.

The predictor-corrector AIMD experiment triggers exact DFT when the residual
passes a threshold; the uncorrected surrogate diverges while correction keeps
the trajectory stable. The paper states that code/data would be published
after acceptance, but no independently verified public production repository
was found in this audit.

### MolGap decision

Use this as a future acceptance/UQ template: a teacher or auxiliary model must
be checked for physics residuals, not only validation MAE. It is not a current
model or data asset.

## 10. WANet/WALoss: eigenstructure-aware Hamiltonian loss at scale

### Primary evidence

- [ICLR 2025 paper](https://arxiv.org/pdf/2502.19227)
- [Microsoft official package containing WANet/SPHNet code](https://github.com/microsoft/sphnet)

### Method and results

WANet targets the scaling failure of elementwise matrix MAE. It introduces
WALoss, which evaluates predicted Hamiltonian blocks in the ground-truth
eigenbasis/overlap and aligns the learned matrix with eigenstructure without
backpropagating through the predicted eigensolver. The paper reports
PubChemQH experiments with B3LYP/Def2-TZV, large molecules, orbital metrics,
system energy, and SCF-cycle behavior.

The reported system-energy MAE on the PubChemQH test set drops from a massive
failure without WALoss to 47.193 kcal/mol with WANet+WALoss; the SCF-cycle
ratio also falls to 82% relative to the baseline. The paper explicitly
diagnoses elementwise MAE as a poor proxy when molecule size and orbital count
grow.

### MolGap decision

WALoss is a useful operator-level loss-design reference and its central warning
is relevant to frontier prediction: matrix MAE and eigenvalue/GAP error can
decouple. The basis, dataset, and explicit Hamiltonian inputs differ from the
active track; no loss or PubChemQH rows are imported.

## 11. SPHNet: adaptive sparsity for Hamiltonian prediction

### Primary evidence

- [ICML 2025 paper](https://proceedings.mlr.press/v267/luo25l.html)
- [arXiv full text](https://arxiv.org/html/2502.01171)
- [official archived repository](https://github.com/microsoft/SPHNet)

### Method and results

SPHNet introduces a Sparse Pair Gate, Sparse Tensor Product Gate, and a
three-phase scheduler that moves from random sparsity to adaptive learned
sparsity and then fixed sparsity. The goal is to remove tensor-product paths
whose contribution is small while preserving the Hamiltonian's equivariant
structure.

The study covers QH9 B3LYP/def2-SVP, MD17 PBE/def2-SVP, and PubChemQH
B3LYP/def2-TZVP. It reports 3.3--4x QH9 speed improvements and scales farther
than QHNet on PubChemQH. Ablations remove roughly 70% of tensor-product
combinations with an explicit speed/accuracy trade-off.

The GitHub repository is public, MIT-licensed, and includes preprocessed
PubChemQH data references, but it is archived/read-only and carries an
explicit use-at-own-risk warning.

### MolGap decision

Adaptive sparsity is an efficiency reference only. It is not a scalar frontier
model, and its largest evidence comes from a different basis and database.
Do not import its data or replace the current GraphState architecture without a
new, explicitly budgeted operator contract.

## 12. QHFlow2: public frontier-aware Hamiltonian checkpoints

### Primary evidence

- [QHFlow2 paper](https://arxiv.org/html/2602.16897)
- [official code](https://github.com/seongsukim-ml/QHFlow2)
- [QH9 checkpoints](https://huggingface.co/ksusu/QHFlow2-QH9)
- [rMD17 checkpoints](https://huggingface.co/ksusu/QHFlow2-rMD17)

### Method and results

QHFlow2 uses an SO(2)-equivariant backbone, a two-stage pair update, and flow
matching over Hamiltonian matrices. It constructs a predicted Hamiltonian from
equivariant pair features, solves the generalized eigenproblem with the
overlap matrix, and evaluates the induced density, energy, forces, HOMO, LUMO,
and Gap. The paper defines the frontier errors directly from the predicted and
reference eigenvalues.

The QH9 experiments use B3LYP/def2-SVP and stable ID/OOD plus dynamic
geometry/molecule splits. The paper reports QHFlow2 frontier metrics alongside
Hamiltonian, occupied-orbital, energy, and force errors, and separately tests
SCF initialization. Its zero-shot PubChemQH analysis is a useful warning:
Hamiltonian MAE remains apparently stable while energy error and SCF behavior
collapse for the largest molecules because orbital ill-conditioning amplifies
small errors near the occupied/virtual boundary.

The official code contains QH9/MD17 configs, dataset sharding, training and
prediction scripts, and checkpoint-loading paths. The QH9 Hugging Face card
exposes small/middle/large/extra-large checkpoints, stable random and size-OOD
splits, and dynamic geometry/molecule splits. The repository also documents
file-level licensing; QH9 checkpoint terms are not the same as source-code
terms and must be checked before redistribution.

### MolGap decision

This is the strongest public model/checkpoint lead in the batch for a frontier
teacher, but it is not an authorized initialization. It requires explicit 3D
coordinates, an overlap matrix, a B3LYP/def2-SVP Hamiltonian basis, and a QH9
training contract. The current PCQM scalar labels cannot reproduce that input
surface, and the project's database-preservation rule excludes QH9 import.

## 13. HELM: Hamiltonian pretraining for low-data energy transfer

### Primary evidence

- [HELM paper](https://arxiv.org/html/2510.00224)
- [OMol25 source project](https://arxiv.org/abs/2505.08762)

### Method and results

HELM uses an equivariant node/edge backbone with a Hamiltonian output head and
a separate scalar-energy head. The two-way protocol is explicit:

1. train the backbone on Hamiltonian matrices;
2. attach a new energy head and either freeze the backbone or fine-tune it;
3. compare both against direct energy training at the same data budget.

The paper introduces OMol_CSH_58k, a 56,657-structure subset of OMol25 with
58 elements, 10--150 atoms, long-range interactions, and def2-TZVPD
Hamiltonian labels. On MD17/QM7 and ∇²DFT, HELM reaches roughly 60 micro-
Hartree Hamiltonian MAE and reports up to 2x low-data energy improvement from
Hamiltonian pretraining. The controlled ∇²DFT tables show direct,
pretrained-frozen, and fine-tuned regimes; the benefit is largest in the
smallest conformer split and remains present on unseen molecules.

The paper currently marks code and data as `TBA`. Therefore the evidence is a
primary-paper method result, not a public code/checkpoint admission.

### MolGap decision

HELM is the clearest evidence that rich electronic supervision can transfer a
representation to a lower-data scalar task. It does not demonstrate direct
PCQM HOMO/LUMO/Gap transfer, and its Hamiltonian labels, basis, elements, and
geometry are external. A future same-database pretraining study would need a
new 6-31G* electronic-label generator or an equivalent public artifact; no
experiment is authorized now.

## 14. QHNetV2: efficient SO(2)-local Hamiltonian architecture

### Primary evidence

- [QHNetV2 paper](https://arxiv.org/html/2506.09398)
- [AIRS code repository](https://github.com/divelab/AIRS)

### Method and results

QHNetV2 uses local SO(2) frames for off-diagonal pair updates and node
features, then maps back to globally SO(3)-equivariant Hamiltonian blocks. It
replaces expensive SO(3) Clebsch--Gordan products with SO(2) operations while
retaining continuous SO(2) tensor products for many-body node fusion.

On QH9-stable-id the paper reports all-Hamiltonian MAE 31.50 micro-Hartree,
orbital-energy MAE 417.89 micro-Hartree, and occupied-wavefunction similarity
98.58%; QH9 dynamic molecule-OOD remains materially harder. On MD17 it
improves Hamiltonian and eigen-energy errors over QHNet, and the efficiency
study reports 4.34x speed over QHNet on QH9 with lower memory than QHNet.

The public implementation is part of AIRS. The QH9/QHNetV2 paper evaluates
Hamiltonian and eigen-energy metrics, not a direct HOMO/LUMO/Gap head.

### MolGap decision

This is a strong computational-efficiency and equivariant-operator reference,
not a current scalar Gap experiment. It still requires explicit 3D geometry,
orbital basis conventions, overlap/Hamiltonian labels, and external QH9/MD17
data. Preserve the current GraphState route unless a separate operator track
is explicitly approved.

## 15. Fully differentiable ML/QM Hamiltonian learning

### Primary evidence

- [JCTC paper/preprint](https://doi.org/10.1021/acs.jctc.5c00522),
  [open arXiv record](https://arxiv.org/abs/2504.01187), and the
  [Atomistic Cookbook implementation](https://atomistic-cookbook.org/examples/hamiltonian-qm7/hamiltonian-qm7.html)
- [PySCFAD code](https://github.com/fishjojo/pyscfad) and
  [documentation](https://fishjojo.github.io/pyscfad/)

### Method and results

Suman et al. separate the model into an effective Hamiltonian predictor and a
differentiable quantum-chemistry layer. A direct Hamiltonian model is first
prefit on a minimal basis; the indirect model then backpropagates through
PySCFAD and optimizes quantities derived from that Hamiltonian. The paper
compares single and multiple targets, including molecular orbital energies,
dipoles, polarizabilities, and Mayer bond order, and asks whether a small
STO-3G-shaped Hamiltonian can reproduce observables computed from a larger
def2-TZVP reference.

The important result is not a single headline Gap number. Adding target
constraints reallocates limited model capacity: the observable whose loss is
added improves, while other observables can lose accuracy. The effective
Hamiltonian can also learn a useful basis-set correction, but basis size and
symmetry remain part of the model contract. The [Cookbook example](https://atomistic-cookbook.org/examples/hamiltonian-qm7/hamiltonian-qm7.html)
is a runnable engineering artifact, not a PCQM benchmark.

### MolGap decision

This is the strongest public route for a future differentiable readout/control
study, but it still needs basis integrals, operator labels, and a 3D contract.
It cannot be attached to the current scalar PCQM rows without changing the
label surface. No PySCFAD model, Hamiltonian, or external QM7/QM9 row is
imported.

## 16. SAP-feature Hamiltonian Delta-learning

### Primary evidence

- [SAP Hamiltonian paper](https://arxiv.org/html/2606.12326)

### Method and results

The SAP model uses a superposition-of-atomic-potentials initial guess as a
low-cost electronic baseline. SAP orbitals define a symmetry-adapted intrinsic
atomic-orbital plus projected-atomic-orbital basis; the model predicts the
converged PBE0 Fock matrix in that basis. The direct model learns a matrix
residual, `Delta F = F_PBE0 - F_SAP`, and adds orbital-energy, density-matrix,
and dipole penalties. Core and valence blocks use separate GNNs, and Fourier
features cover multiple length/energy scales.

On QM9, the direct PBE0/cc-pVDZ model reaches at 16,000 training molecules
about `21.2 meV` HOMO MAE, `21.2 meV` LUMO MAE, and `28.4 meV` Gap MAE. A
downfolded cc-pVTZ model trained to 32,000 examples reports `40.1/41.9/55.2
meV` for HOMO/LUMO/Gap; the paper explicitly attributes the degradation to
basis-set mismatch and does not use the same Delta parameterization for the
downfolded target. The paper reports no official training repository or
checkpoint.

### MolGap decision

This is the clearest evidence that a physically meaningful low-fidelity
Hamiltonian can make an operator residual easier to learn, and that the
residual must be defined in the same basis. It is not evidence that an xTB or
SAP scalar can be subtracted from the current B3LYP/6-31G* Gap. Any future
scalar Delta test must define a cheap scalar baseline and its coordinate/label
identity separately; no SAP features or QM9 rows are admitted.

## 17. Direct one-particle density-matrix prediction as an SCF teacher

### Primary evidence

- [JCTC paper](https://doi.org/10.1021/acs.jctc.4c00042)
- [open arXiv record](https://arxiv.org/abs/2401.06533) and
  [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11171273/)

### Method and results

Hazra, Patil, and Sanvito directly predict every independent element of a
Kohn--Sham one-particle density matrix from structural coordinates, rather than
predicting a scalar property or projecting a real-space density into a basis.
The study uses PySCF BLYP/cc-pVDZ data for H2O, S2O, and a metal--water test
system. Each model uses 9,000 train, 800 validation, and 1,000 test structures;
the molecular trajectories are generated by Born--Oppenheimer MD or controlled
rigid displacements. The learned density is evaluated as an SCF initial guess
and, without a self-consistent cycle, for energies and forces. The reported
test density-matrix fits are around `2e-4 au` MAE for the small systems and
`R^2` above `0.999`, but the matrix dimension scales quadratically with the
basis size and idempotency is not enforced.

### MolGap decision

This is a useful negative/teacher control: a low matrix MAE is not enough; SCF
iterations, energies, forces, and physical constraints must also be checked.
Its molecule-specific coordinates, BLYP/cc-pVDZ theory, small systems, and no
public production checkpoint exclude direct use for PCQM/ETKDG. No density
matrix or coordinate archive is imported.

## 18. HamGNN: completed equivariant Hamiltonian artifact

### Primary evidence

- [npj Computational Materials paper](https://doi.org/10.1038/s41524-023-01130-4)
- [official code](https://github.com/QuantumLab-ZY/HamGNN)
- [Zenodo pretrained models](https://doi.org/10.5281/zenodo.8147631) and
  [Zenodo training data](https://doi.org/10.5281/zenodo.8157128)

HamGNN is an E(3)-equivariant graph model for ab-initio tight-binding/
Hamiltonian matrices across molecules and solids. The public package includes
Hamiltonian reconstruction and band/eigenvalue post-processing, while the
paper reports QM9 molecular tests and transfer to crystalline, defect, and
heterostructure systems. The Zenodo records make this a completed operator
artifact rather than a paper-only claim.

### MolGap decision

HamGNN is useful for code layout, equivariant block parameterization, and
operator-to-spectrum validation. Its basis/matrix labels, tight-binding and
materials transfer roles, explicit 3D inputs, and GPL-3.0 code/data surface do
not match the current scalar B3LYP/6-31G*/ETKDG contract. No code, checkpoint,
or Zenodo data is imported.

## 19. HAMSTER: physics-informed Hamiltonian Delta-learning

### Primary evidence

- [Nature Communications paper](https://doi.org/10.1038/s41467-026-70865-7) and
  [arXiv record](https://arxiv.org/abs/2508.20536)
- [Hamster.jl code](https://github.com/TheoFEM-TUM/Hamster.jl)
- [Zenodo data release](https://doi.org/10.5281/zenodo.18485403)

HAMSTER learns the difference between a physically motivated tight-binding
Hamiltonian and a DFT Hamiltonian, rather than fitting the Hamiltonian from
scratch. The public Julia package separates TB, ML, and spin--orbit blocks and
the data release contains GaAs, CsPbBr3, and MAPbBr3 reference/input/output
archives with checksums. For CsPbBr3, the paper reports eigenvalue MAEs of
`0.047`, `0.049`, and `0.057 eV` at 425/525/625 K and band-gap differences below
`50 meV`; the workflow then scales to a 20,480-atom supercell. The Delta
baseline is periodic PBE with explicit SOC where needed, not a molecular
semiempirical orbital label.

### MolGap decision

HAMSTER is a strong completed Delta-learning and artifact-packaging reference,
especially for logging a low-fidelity baseline, residual target, physical
post-processing, and large-system acceptance. Its periodic inorganic materials,
PBE/SOC/VASP/TB contract and 188-GB data release make it an external method
reference. It does not authorize a current database change or a scalar Gap run.

## Cross-paper synthesis

| Evidence family | What is supported | What is not supported for MolGap |
|---|---|---|
| Public operator benchmark | QH9 and nablaDFT/∇²DFT have real loaders, splits, labels, and checkpoints | They do not become B3LYP/6-31G*/ETKDG labels by renaming fields |
| Self-consistency | Hamiltonian/density residuals can provide unlabeled objectives and label-free confidence | A scalar PCQM row has no H/S/density/integral object for these losses |
| Fixed-point / DEQ | Iterating a learned correction can improve stability and transfer | Fixed-point operator training is not scalar Gap Delta-learning |
| Frontier-aware readout | Eigenvalue-weighted objectives expose the mismatch between matrix MAE and Gap error | Basis-dependent eigensolvers cannot be attached to the current scalar database without new labels |
| Hamiltonian pretraining | Rich matrix supervision can transfer to low-data energy heads | No same-contract PCQM frontier transfer or public 6-31G* Hamiltonian teacher was verified |
| Public checkpoints | QHFlow2 and NeuralSCF/HamEvo expose real artifacts | External weights cannot enter the random-init screen or bypass ETKDG/target-contract gates |
| Differentiable ML/QM layer | PySCFAD makes Hamiltonian-to-orbital/response losses and multi-target tradeoffs explicit | The current scalar database has no Hamiltonian, overlap, density, or integral labels |
| Operator Delta-learning | SAP and HAMSTER show that `Delta F`/`Delta H` can use a physics baseline and improve data efficiency | Operator residuals cannot be renamed `Delta Gap`; the baseline must share basis, theory, geometry, and identity |
| Density teacher | Direct density-matrix prediction can improve SCF initialization and non-self-consistent observables | Matrix dimension, idempotency, coordinate alignment, and external theory make it a teacher-design control only |
| Completed operator artifacts | HamGNN, HamEvo, QHFlow2, and HAMSTER expose code/data/checkpoint patterns | None provides a same-contract PCQM B3LYP/6-31G*/ETKDG frontier model |

## Evidence-only follow-up gate

The following are hypotheses, not authorized experiments:

1. **Same-database electronic teacher:** require a train-only, immutable
   B3LYP/6-31G* electronic-label manifest aligned to existing PCQM identities,
   plus no-teacher and random-init controls. External QH9/∇²DFT rows do not
   satisfy this.
2. **Scalar Delta control:** define `Delta Gap` against a cheap baseline whose
   geometry, target definition, and leakage are fully auditable. Do not call
   `Delta H` or a low-fidelity orbital residual a scalar Delta result.
3. **Teacher acceptance:** require frontier MAE, algebraic Gap consistency,
   OOD residuals, and an on-manifold/SCF-style failure check. A lower auxiliary
   loss alone is insufficient.
4. **Cost gate:** keep all parsing, geometry, and label construction on CPU;
   reserve accelerators for a frozen accepted cache and a paired seed-42
   screen. No remote job is authorized by this record.

## Primary-source index

- [QH9 paper](https://arxiv.org/html/2306.09549), [AIRS/QHBench](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9), and [Zenodo](https://zenodo.org/records/8274793)
- [Self-Consistency Training](https://proceedings.mlr.press/v235/zhang24ak.html) and [arXiv](https://arxiv.org/html/2403.09560)
- [DEQHNet](https://proceedings.neurips.cc/paper_files/paper/2024/file/a31d1942f4f4f52a73bfd1d67856bed7-Paper-Conference.pdf) and [code](https://github.com/Zun-Wang/DEQHNet)
- [QHFlow](https://arxiv.org/html/2505.18817) and [code](https://github.com/seongsukim-ml/QHFlow)
- [nablaDFT](https://doi.org/10.1039/D2CP03966D) and [repository](https://github.com/AIRI-Institute/nablaDFT)
- [∇²DFT](https://arxiv.org/html/2406.14347) and [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/40d45b1e23d00d5895e65778e85cf8ee-Paper-Datasets_and_Benchmarks_Track.pdf)
- [NeuralSCF](https://doi.org/10.1038/s41524-026-02110-0), [code](https://github.com/songfeitong/neuralscf), and [data](https://doi.org/10.57760/sciencedb.37834)
- [HamEvo](https://arxiv.org/html/2606.14498), [code](https://github.com/axdfhj/HamEvo_official), and [data](https://huggingface.co/datasets/ZJUSCL/hamevo-data)
- [Self-consistent validation](https://arxiv.org/html/2402.10186)
- [WANet/WALoss](https://arxiv.org/pdf/2502.19227) and [SPHNet package](https://github.com/microsoft/sphnet)
- [SPHNet](https://proceedings.mlr.press/v267/luo25l.html) and [archived code](https://github.com/microsoft/SPHNet)
- [QHFlow2](https://arxiv.org/html/2602.16897), [code](https://github.com/seongsukim-ml/QHFlow2), and [QH9 checkpoints](https://huggingface.co/ksusu/QHFlow2-QH9)
- [HELM](https://arxiv.org/html/2510.00224) and [OMol25](https://arxiv.org/abs/2505.08762)
- [QHNetV2](https://arxiv.org/html/2506.09398) and [AIRS](https://github.com/divelab/AIRS)
- [Fully differentiable ML/QM Hamiltonian learning](https://doi.org/10.1021/acs.jctc.5c00522), [arXiv](https://arxiv.org/abs/2504.01187), [Atomistic Cookbook](https://atomistic-cookbook.org/examples/hamiltonian-qm7/hamiltonian-qm7.html), and [PySCFAD](https://github.com/fishjojo/pyscfad)
- [SAP-feature Hamiltonian learning](https://arxiv.org/html/2606.12326)
- [One-particle density matrix prediction](https://doi.org/10.1021/acs.jctc.4c00042), [arXiv](https://arxiv.org/abs/2401.06533), and [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11171273/)
- [HamGNN](https://doi.org/10.1038/s41524-023-01130-4), [code](https://github.com/QuantumLab-ZY/HamGNN), [models](https://doi.org/10.5281/zenodo.8147631), and [data](https://doi.org/10.5281/zenodo.8157128)
- [HAMSTER](https://doi.org/10.1038/s41467-026-70865-7), [code](https://github.com/TheoFEM-TUM/Hamster.jl), and [data](https://doi.org/10.5281/zenodo.18485403)
