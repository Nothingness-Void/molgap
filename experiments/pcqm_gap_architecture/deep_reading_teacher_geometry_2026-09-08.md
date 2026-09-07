# Deep Reading: Teacher Distillation and Conformer-Aligned Geometry Pretraining

Date: 2026-09-08

This note records two additional primary-source reads at the intersection of
pretraining, teacher models, geometry, and deployment-time simplification:
PG-MLD and ConforFormer. They were selected because both expose a concrete
teacher signal and a reproducible training objective, but they solve different
problems. PG-MLD transfers dynamic 3D and local-electronic information into a
SMILES student. ConforFormer makes representations of multiple conformers of
the same molecule agree before a frozen embedding is used downstream.

This is an evidence record, not an experiment protocol. It authorizes no
download, external-weight initialization, database merge, remote job, or
production change. The permitted databases remain official PCQM4Mv2 for Track
B and the repaired-2M PubChemQC corpus for Track A. Any later adaptation must
use train-role molecules only, keep official validation/test-dev sealed, and
use ETKDG consistently for every geometry view. The source papers' ZINC,
Uni-Mol, OMol, OpenMolecules, OpenMM, charge, and other external assets are
not converted into MolGap training data by this note.

## Evidence gate and disposition

| Source | Primary evidence | Artifact evidence | Direct PCQM Gap evidence | MolGap disposition |
|---|---|---|---|---|
| PG-MLD | Full bioRxiv preprint with teacher, distillation, MoleculeNet tables, and cross-student ablations | Public GitHub repository exposes loaders, models, configs, training entry points, and atomic checkpoint code; large data, trajectories, language-model weights, and checkpoints are explicitly external | None; the paper evaluates MoleculeNet and does not report a matched PCQM4Mv2 HOMO/LUMO/Gap run | Dynamic-teacher method reference only; lowest-priority future teacher hypothesis after an ETKDG/PCQM rewrite and explicit 2D-student authorization |
| ConforFormer | Published Digital Discovery article with conformer-alignment objective, Uni-Mol/OMol training protocol, frozen-probe evaluation, and PharmaIsomer benchmark | Public MIT repository contains the Uni-Mol fork, data pipelines, training/inference scripts, checked-in baseline results, and model-weight link; HF exposes a named model file | None; no direct PCQM4Mv2 Gap result is reported | High-quality conformer-invariance and frozen-probe reference; possible post-selection ETKDG teacher adaptation, but no external geometry or weight import |

The evidence is sufficient to admit two method hypotheses, not a claim that
either method will improve PCQM Gap. The distinction matters: a public
implementation closes an artifact question, while the absent same-database
target result leaves the MolGap causal question open.

## 1. PG-MLD: dynamic 3D trajectory knowledge distilled into a SMILES student

### 1.1 What the primary paper actually proposes

[PG-MLD](https://www.biorxiv.org/content/10.64898/2026.07.29.741404v1.full)
uses a two-stage teacher--student design. In the first stage, a 3D teacher
sees a dynamic physical view built from an initial conformer, coordinate
Gaussian noise (CGN), conformational-angle noise (CAN), temporally ordered
molecular-dynamics frames, and atom-level formal/Gasteiger charge features.
The paper's stated goal is to make the teacher represent geometry,
conformational evolution, and local electronic environment together.

The teacher has four information-flow steps:

1. construct the multi-frame physical view;
2. encode each frame with an equivariant geometric network;
3. process the frame sequence with a Liquid Time-Constant (LTC) network; and
4. produce atom-level and molecule-level dynamic representations.

The second stage freezes the teacher and trains a molecular-language student
from SMILES. A token--atom mapping aggregates token features into atom
features, then into a molecule feature. Separate projection heads align the
teacher and student at atom and molecule levels. A cross-modal contrastive
loss aligns the same molecule across modalities; masked language modeling is
retained for student architectures that support it. After distillation, the
teacher is discarded and inference uses only SMILES.

The important scientific separation is therefore:

```text
dynamic 3D trajectory + electronic descriptors
                -> frozen teacher
SMILES -------------------------------> student representation
                -> downstream property head
```

This is not a new random-init graph architecture. It is privileged-information
transfer: the student is allowed to receive information during training that
will not be available at inference.

### 1.2 Training objectives and reported evidence

The primary paper describes teacher-side denoising, distance, charge,
temporal-prediction, and trajectory-consistency objectives. During
distillation it uses atom-level and molecule-level representation alignment,
cross-modal contrastive learning, and optional MLM. The teacher is frozen
while the student learns; this is a useful control boundary because it keeps
the source representation from changing to chase each downstream task.

The paper evaluates three language-model students—ChemBERTa, ChemBERTa-2,
and MoLFormer—on nine MoleculeNet tasks with five runs using seeds 42--46.
It reports the best classification result on all six listed classification
tasks and the lowest RMSE for ESOL and FreeSolv among the compared methods.
The reported values are ROC-AUC `0.966/0.936/0.997/0.851/0.744/0.825` on
BBBP/BACE/ClinTox/Tox21/ToxCast/HIV and RMSE `0.627/0.945` on ESOL/FreeSolv.
The cross-student table reports improvement in 25 of 27 model--task settings;
the paper gives ChemBERTa BACE `0.901 -> 0.927` and MoLFormer Lipophilicity
RMSE `0.662 -> 0.620` as representative gains, with two small exceptions.

These are useful teacher-transfer controls, but they do not establish a
PCQM4Mv2 Gap gain. MoleculeNet labels, the external dynamic data role, and
the SMILES student are all different from the current Track B question.

### 1.3 What the public repository confirms

The [official PG-MLD repository](https://github.com/lzh23399/PG-MLD) is public
and exposes the data, model, training, configuration, and visualization
surfaces. Its README explicitly says that molecular datasets and MD
trajectories, pretrained language-model weights, and PG-MLD checkpoints are
not stored in the repository; they must be obtained or generated separately.
That is a material reproducibility distinction from a repository that ships a
self-contained checkpoint.

The audited teacher configuration uses the relative path
`zinc/zinc15_250K_openmm_full`, batch size 4, at most 8 frames, 384-dimensional
teacher states, 4 geometric layers, 8 attention heads, 16 radial basis
channels, a 10.0 cutoff, `electronic_dim=2`, and coordinate-update scale 0.1.
The visible noise and loss settings are:

| Teacher setting | Repository value |
|---|---:|
| extra coordinate Gaussian noise | `sigma=0.01` |
| electronic-channel mask probability | `0.15` |
| denoising loss weight | `1.0` |
| distance loss weight | `0.1` |
| charge loss weight | `0.25` |
| temporal loss weight | `0.5` |
| consistency loss weight | `0.25` |
| epochs / learning rate | `10 / 2e-4` |
| optimizer regularization | weight decay `0.01`, gradient clip `5.0` |
| random seed | `2026` |

The distillation configuration uses the teacher checkpoint
`latest.pt`, a ChemBLM student, batch size 32, at most 8 randomly cropped
frames, atom and molecule alignment weights of `1.0`, contrastive weight
`0.05`, temperature `0.2`, and MLM weight `0.0` in the visible ChemBLM
configuration. The repository's configuration therefore makes the role of
the optional MLM objective inspectable rather than assuming that every
student uses the same loss.

The loader performs additional contract work that is relevant to MolGap
engineering: it keeps heavy atoms, aligns trajectory atom order to an RDKit
SMILES-derived order, validates the `[T,N,3]` trajectory shape, supports
sharded streaming, and writes `latest.pt` through a temporary path before
replacement. Those are good durability patterns. They do not solve the
scientific mismatch: the visible data path is external OpenMM/ZINC data, the
electronic channels are formal/Gasteiger charges, and the geometry is not the
project's ETKDG-only lineage.

The public repository currently exposes one commit and no clearly surfaced
license file in the audited GitHub view. Consequently its code is a method
reference, not a drop-in dependency. The paper is a bioRxiv preprint; its
headline numbers should be treated as primary preprint evidence rather than
peer-reviewed PCQM evidence.

### 1.4 MolGap interpretation

The positive idea is not “use OpenMM trajectories.” It is the separation of
privileged teacher information from the deployed student, with explicit atom
and molecule alignment and a frozen-teacher boundary. A database-preserving
adaptation, if later authorized, would have to change all of the following:

- replace the external dynamic-MD source with conformers generated only from
  permitted PCQM train-role SMILES;
- use ETKDG for every geometry view, including the initial and perturbed
  views, and record failed conformers/fallbacks;
- remove or separately declare formal/Gasteiger channels rather than calling
  them quantum supervision;
- use only train-role source tasks and labels, with no official validation or
  test-dev exposure;
- compare a fresh random-init 2D student, a teacher-free geometry objective,
  and the frozen-teacher route under equal optimization and parameter budgets;
- keep the teacher and student artifact manifests separate, with teacher
  checkpoint hash, projection-head hash, and student-inference hash.

Even after those changes, the route would be a post-selection teacher study,
not a valid addition to the active random-initialized architecture screen.
The dynamic trajectory and continuous-time components introduce substantial
implementation and compute risk without direct PCQM Gap evidence.

**Disposition: B/C.** Keep PG-MLD as a complete method/code reading and a
low-priority future teacher hypothesis. Do not import its ZINC/OpenMM data,
charges, checkpoint, or student weights. Do not use its MoleculeNet metrics as
MolGap evidence.

## 2. ConforFormer: conformer identity as a contrastive invariance target

### 2.1 The primary objective

The published [ConforFormer article](https://doi.org/10.1039/D6DD00096G)
frames a molecule as an identity with several 3D conformers. Two distinct
conformers of the same molecule are positive pairs; conformers from different
molecules are negatives. The input to the model is atomic numbers and 3D
coordinates. The molecular graph is used when constructing the pair labels,
but it is not supplied as an input to the encoder.

The method adds a normalized temperature-scaled cross-entropy (NT-Xent)
objective to the original Uni-Mol masked-token, coordinate-denoising, and
masked-distance objectives. The reported implementation uses embedding
dimension `d=512`, `n=128` unique molecules per batch, two conformer views per
molecule, and temperature `tau=0.07`. The contrastive term is therefore a
clean identity-invariance target, not an ambiguous “3D augmentation” claim:
the positive relation is defined by molecular identity, while the input view
varies by conformer.

The paper trains ConforFormer variants on the Uni-Mol conformer corpus and on
an OMol subset. The Uni-Mol corpus uses ten conformations per molecule
generated with RDKit and optimized with MMFF94. The OMol model uses a higher-
quality quantum-chemical geometry subset described by the paper. Frozen CLS
embeddings are evaluated with lightweight downstream models on MoleculeNet;
the article reports ConforFormer-OMol as best on four of six quantum-regression
tasks and five of eight classification tasks, with some task-specific failures.
The PharmaIsomer benchmark is a separate structural test: at 50% recall, the
reported precision rises from roughly 8% for a Uni-Mol replicate to above 83%
for ConforFormer-OMol, while a backbone/isomer precision is reported at 94%.

The strongest transferable evidence is not the absolute MoleculeNet table. It
is the controlled statement that the same molecular identity should remain
nearby in representation space across accepted conformers, and that frozen
embeddings are evaluated separately from a fully fine-tuned downstream model.

### 2.2 Repository and code audit

The [official ConforFormer repository](https://github.com/EPiCs-group/ConforFormer)
is public and MIT-licensed. It contains a Uni-Mol research fork, data
processing pipelines, training/fine-tuning/inference scripts, baseline
results, and instructions for generating the conformer and isomer assets. The
repository links named model weights on
[Hugging Face](https://huggingface.co/ConforFormer/ConforFormer), but no weight
was downloaded or initialized in MolGap.

The checked-in contrastive training script makes the full objective and cost
surface concrete. It uses one GPU, batch size 128, learning rate `5e-4`,
weight decay `1e-4`, 5,000 warm-up steps, 2,000,000 updates, 15 encoder layers,
512-dimensional embeddings, 2,048-dimensional feed-forward layers, and 64
attention heads. The visible objective weights are masked-token `1`, masked-
coordinate `5`, masked-distance `10`, contrastive `2`, x-norm `0.01`, and
delta-pair-representation norm `0.01`; the contrastive temperature is `0.07`.
The script sets seed `0`, uniform coordinate noise of `1.0`, and mask
probability `0.15`.

The conformer sampler is deterministic conditional on the dataset, epoch,
index, and seed: it selects two distinct conformers without replacement when
two or more are present, and duplicates the only conformer when a molecule
has only one. This is a useful implementation detail for a future pair-manifest
audit because it makes the positive-pair policy explicit. It also exposes a
failure mode: duplicating a single conformer creates a positive pair with no
geometric variation, so the fraction of one-conformer molecules must be
reported rather than hidden.

The loss implementation normalizes the two CLS embeddings, forms an in-batch
InfoNCE matrix, and adds the contrastive loss to the original Uni-Mol losses.
One code-quality caveat is material for reproducibility: the visible loss file
contains hard-coded distance normalization constants with comments saying
they are random numbers and that their source is unknown. The line does not
invalidate the scientific paper, but it means a local reproduction must record
the exact repository revision and either verify or independently recompute
those constants before treating a run as a faithful reproduction.

### 2.3 MolGap interpretation

ConforFormer is more portable than PG-MLD at the objective level because it
does not require trajectories, a continuous-time recurrent state, or an
external electronic descriptor. A permitted adaptation could ask whether two
accepted ETKDG conformers of the same train-role molecule should produce
similar intermediate representations. That adaptation would still be a new
geometry/pretraining contract, not a free improvement to GraphState.

The required constraints are:

1. generate both views with the same pinned ETKDG method used by MolGap;
2. pair only identical canonical train-role molecules and never cross a sealed
   role boundary;
3. record the number of molecules with one, two, and more accepted conformers;
4. compare identity positives against hard negatives matched on formula,
   scaffold, or fingerprint similarity, with the negative construction fixed
   before evaluation;
5. compare scratch, frozen-probe, fine-tuned, and no-contrastive controls;
6. keep the deployed prediction path explicit: either one deterministic ETKDG
   conformer, a deterministic conformer pool, or a graph-only student; and
7. report representation invariance and downstream Gap separately, since an
   invariance gain can coexist with a regression error or bias increase.

The adaptation should not import Uni-Mol/OMol conformers or the ConforFormer
checkpoint. It should not claim that a frozen-probe result is an end-to-end
gain without a matched scratch/fine-tune control. The paper's strongest
structural benchmark, PharmaIsomer, is not a PCQM Gap target and cannot be
used as evidence that the project will improve B3LYP/6-31G* frontier levels.

**Disposition: A/B as an external reference, B as a future MolGap method.**
The published paper and MIT code close the method and artifact surface more
cleanly than PG-MLD. Keep it as a high-quality post-selection geometry-teacher
and frozen-probe reference. No external conformers, OMol rows, model weights,
or Uni-Mol data are admitted.

## 3. Cross-paper decision under the fixed MolGap database

| Question | PG-MLD | ConforFormer | Decision for MolGap |
|---|---|---|---|
| What is the teacher signal? | Dynamic 3D trajectory, perturbation, and local charge representation | Same-identity conformer embedding agreement | Prefer the simpler identity-invariance hypothesis first if a later geometry study is authorized |
| Is the source task target-aligned? | No; MoleculeNet downstream and external dynamic data | No; MoleculeNet/PharmaIsomer downstream and external conformer corpora | Neither is a direct PCQM Gap result |
| Does inference require privileged geometry? | No after distillation; student uses SMILES | Frozen embedding workflow uses 3D; a graph/SMILES student is not the published deployment | Both require a new deployment decision |
| Is the geometry contract compatible? | No: OpenMM/CGN/CAN/external trajectories | No: RDKit/MMFF/OMol rather than ETKDG | Rewrite geometry before any possible test |
| Artifact strength | Inspectable code/config, no bundled data or weights, unclear license surface | MIT code, data pipelines, results, and named HF weights | ConforFormer has the stronger reproduction surface; neither gets imported |
| Main risk | Large dynamic teacher and extra charge/trajectory contracts | Positive-pair quality and conformer-pool coverage | Start with CPU identity/pair acceptance, not GPU training |

The combined evidence does not justify reopening the active random-init
architecture screen. It does support two narrowly separated post-selection
questions:

- **M11, simpler geometry teacher:** ETKDG conformer-alignment pretraining or
  frozen-probe, inspired by ConforFormer, with a graph-state scratch control.
- **M12, richer privileged teacher:** a PCQM/ETKDG-only, non-trajectory rewrite
  of the PG-MLD atom/molecule alignment idea, only if a 2D student route is
  explicitly authorized and its extra complexity is budgeted.

M11 and M12 must not be stacked. A positive result from M11 would not prove
that dynamic distillation helps; a positive result from M12 would not prove
that conformer-invariance was the cause. Each needs its own source/target
manifest, checksum, control, and decision record.

## 4. Unadmitted source-task lead

Search results also surfaced a 2026 item titled
“Supervised Source-Task Pretraining for Low-Data Electrochemical Molecular
Property Prediction.” The available page during this audit was a ResearchGate
metadata/abstract record, not a primary full text with a verified official
implementation. The abstract suggests QM9 pretraining on HOMO, LUMO, Gap, and
`u0` followed by transfer to electrochemical tasks. Because the primary paper,
exact roles, geometry, split, and code were not closed, it is not counted in
the literature ledger and is not a possible experiment. It remains a search
lead only. A reliable primary source must be retrieved before any claim is
made about it.

## Reproducibility checklist before any future protocol

1. Pin the primary paper version, repository commit, and model/config path.
2. Record license, file size, SHA-256, and whether each data or checkpoint
   artifact is actually bundled or only linked.
3. Build a canonical train-role identity manifest before generating pairs,
   conformers, auxiliary labels, or teacher outputs.
4. Use ETKDG for every MolGap geometry view and record seeds, failures,
   fallback behavior, conformer count, and atom ordering.
5. For a teacher route, freeze the teacher and store teacher/student/projector
   manifests separately; never let the student update the teacher silently.
6. Compare scratch, no-teacher/no-contrastive, frozen-probe, and fine-tuned
   controls under equal molecule, optimizer, parameter, and accelerator
   budgets.
7. Keep Gap as the scientific target. Any HOMO/LUMO, charge, or descriptor
   source task must be declared as a train-only auxiliary role and checked for
   role leakage.
8. Report representation diagnostics separately from official Gap metrics,
   including conformer invariance, hard-negative separation, calibration, and
   error by molecule size/scaffold/conjugation regime.
9. Require a material paired seed-42 gain before requesting seeds 43/44, and
   do not allocate a remote job until the CPU pair/geometry cache passes
   acceptance.
10. Preserve atomic checkpoints and independently retrievable output chunks;
    no transient teacher worker or single long-running task may be the only
    copy of evidence.

## Primary sources

- [PG-MLD bioRxiv full text](https://www.biorxiv.org/content/10.64898/2026.07.29.741404v1.full) and [official code](https://github.com/lzh23399/PG-MLD)
- [ConforFormer Digital Discovery article](https://doi.org/10.1039/D6DD00096G), [indexed full article](https://www.sciencedirect.com/org/science/article/pii/S2635098X26001397), [official MIT repository](https://github.com/EPiCs-group/ConforFormer), and [model card](https://huggingface.co/ConforFormer/ConforFormer)
- [Unadmitted source-task lead](https://www.researchgate.net/publication/405238198_Supervised_Source-Task_Pretraining_for_Low-Data_Electrochemical_Molecular_Property_Prediction) (metadata/abstract only; not counted)
