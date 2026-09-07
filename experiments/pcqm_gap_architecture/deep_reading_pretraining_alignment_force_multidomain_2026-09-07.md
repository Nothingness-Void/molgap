# Deep Reading: Pretraining Evidence, Force-Centric Objectives, and Data Alignment

This record extends the auditable literature reserve with four sources that are
directly relevant to pretraining decisions but do not authorize a new MolGap
run. The project database remains the official PCQM4Mv2 route and the repaired
2M PubChemQC route. No external rows, checkpoint, geometry, or warm start was
imported.

## Evidence boundary

The four sources answer different questions and must not be collapsed into one
claim:

| Source | What is actually evidenced | What is not evidenced for MolGap |
|---|---|---|
| [Does GNN Pretraining Help Molecular Representation?](https://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html) | Broad controlled ablations and a pretraining negative-control lesson | No PCQM4Mv2 HOMO/LUMO/Gap result or current-contract artifact |
| [ET-OREO / May the Force be with You](https://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html) | Force-centric 3D pretraining across equilibrium and off-equilibrium conformations | No ETKDG-only train/inference route, no direct PCQM Gap result, and no verified code/checkpoint in this audit |
| [JMP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a896a8bb065774169d9ad65f19208b7-Abstract-Conference.html) | Large supervised multi-domain pretraining with public code/config/checkpoint links and broad downstream evaluation | External 3D theories/data, no direct PCQM Gap comparison, and a non-commercial repository license |
| [CSI / data alignment](https://arxiv.org/abs/2502.11085) | Task-aligned upstream selection can match or beat large mixed pretraining at a reported 1/24 budget; public selection scripts exist | No PCQM4Mv2 Gap experiment, no same-database result, and no evidence that CSI transfers unchanged to this graph/target contract |

The evidence grade used here is deliberately conservative:

- **A:** primary paper plus a public, inspectable implementation or asset
  surface with enough information to reproduce the reported class of result;
- **B:** primary paper gives a coherent method/result, but the target, geometry,
  artifact, or task contract is not MolGap-compatible;
- **C:** observation, incomplete artifact, or negative control that is useful
  for search direction but cannot support an experiment claim.

## 1. Does GNN Pretraining Help Molecular Representation?

### Primary evidence

The NeurIPS 2022 paper explicitly studies whether molecular GNN pretraining
benefits survive controlled choices of objective, split, input features,
pretraining scale, and GNN architecture. The official abstract states three
important findings: self-supervised pretraining is not consistently
statistically better than scratch; supervised pretraining can help but the
gain shrinks with richer features or more balanced splits; and
hyperparameters can matter more than the pretraining objective on small
downstream data. The paper studies ZINC15-scale self-supervision, a much larger
SAVI pretraining source, and supervised ChEMBL-style pretraining against
MoleculeNet tasks; it does not run the current PCQM4Mv2 Gap contract.

### What this changes for MolGap

This is a negative/control reference, not an argument against all pretraining.
It changes the minimum design of any future same-database test:

1. scratch and pretrained models must use the same encoder, features, optimizer
   family, internal split, and stopping rule;
2. pretraining-only, frozen-probe, and end-to-end fine-tuning results must be
   separated;
3. an input-feature ablation is required because a richer random-init feature
   set can erase an apparent pretraining advantage;
4. pretraining scale and objective should not be varied together in the first
   screen.

The result is especially relevant because MolGap already has a strong
random-init GraphState route. A pretrained representation that only improves a
linear probe, or only wins after a new feature/optimizer choice, is not a
production improvement.

### Artifact and contract status

The primary NeurIPS record and paper are retrievable. A verified author-
maintained code/checkpoint package was not located in this audit. The source is
therefore **B/C mechanism and negative-control evidence**, not an executable
candidate.

## 2. ET-OREO: force-centric 3D pretraining

### Primary method

The NeurIPS 2023 paper proposes ET-OREO, a force-centric pretraining model for
equilibrium and off-equilibrium molecular conformations. Its central distinction
is between the two data regimes:

- equilibrium structures receive zero-force regularization and force-guided
  denoising;
- off-equilibrium structures receive direct force supervision;
- forces are obtained as energy gradients, allowing one objective to cover both
  regimes.

The paper reports a unified pretraining set of more than 15M conformations,
assembled from PCQM4Mv2 equilibrium conformations together with ANI1x, MD17,
and Poly24 off-equilibrium sources. It reports roughly a three-fold force
accuracy improvement over an unpretrained Equivariant Transformer and faster
MD inference than NequIP, while property transfer is described as competitive
with state of the art. These are force/MD and general property-transfer claims,
not a direct PCQM4Mv2 HOMO/LUMO/Gap score.

The paper's configuration is a large equivariant 3D route: an Equivariant
Transformer/TorchMD-Net style encoder with scalar/vector geometry, force-derived
losses, noisy 3D coordinates, and multi-corpus conformational supervision. The
training coordinates are not the project's ETKDG-only construction.

### Transferable idea and hard incompatibilities

The useful idea is not to copy the whole force field. It is the distinction
between a clean geometry embedding and a perturbed geometry target, plus an
explicit consistency signal for equilibrium structures. A future same-database
adaptation could ask whether a cheap ETKDG perturbation objective improves the
existing graph encoder, but that would be a new pretraining protocol, not a
drop-in checkpoint import.

The incompatibilities are decisive for the current route:

- MolGap has scalar frontier targets, not atomic forces or energy gradients;
- the accepted training and inference coordinates must both be ETKDG;
- the published force labels and external conformer sources introduce new
  databases and theory/geometry roles;
- no author-maintained code/checkpoint/data package was independently closed in
  the present audit.

**Disposition:** **B method reference, no current experiment.** It raises the
priority of an ETKDG-only denoising/control protocol only after architecture
selection, but it does not justify force labels, external conformers, or a new
database.

## 3. JMP: large supervised multi-domain pretraining

### Primary paper evidence

JMP (Joint Multi-domain Pre-training) trains a shared GemNet-OC-style model as
multiple supervised tasks over approximately 120M systems from OC20, OC22,
ANI-1x, and Transition-1x. The ICLR paper reports an average improvement of
59% over scratch and state-of-the-art or matched performance on 34 of 40
downstream tasks, including QM9, rMD17, MatBench, QMOF, SPICE, and MD22. It
also reports a 235M-parameter model and more than 12x faster fine-tuning after
the upfront pretraining cost.

These numbers are broad cross-domain atomic-property results. They are not a
matched PCQM4Mv2 Gap result: the pretraining sources, labels, coordinates,
energy/force tasks, and downstream contracts differ from B3LYP/6-31G* scalar
frontier prediction.

### Public code and asset audit

The [official repository](https://github.com/facebookresearch/JMP) is publicly
inspectable and exposes configs, preprocessing scripts, fine-tuning paths, and
named `JMP-S`/`JMP-L` checkpoint links. Its README says the data are not stored
in the repository because of size, gives download/preprocessing instructions,
and states that the repository is deprecated/archived while functionality is
being integrated into Open Catalyst. The README also states that the majority
of the project is CC-BY-NC, with separate licenses for dependencies and some
components. This is strong engineering evidence but not a clean production
dependency for MolGap.

### What can be borrowed without changing the database

The portable parts are the protocol and bookkeeping:

- treat each upstream corpus as a named task rather than silently concatenating
  incompatible labels;
- maintain per-task normalization, sampling, and validation metrics;
- separate pretraining cost from fine-tuning cost;
- retain an explicit scratch control and a frozen-trunk/readout-only control;
- record the license of both the code and any downloaded checkpoint.

The full JMP model, OC20/OC22/ANI-1x/Transition-1x rows, and checkpoint cannot
enter the current database or initialization boundary. **Disposition:** **A/B
external pretraining/configuration reference; no current experiment.**

## 4. CSI: pretraining-data alignment over scale

### Primary claim

The TMLR 2026 paper, titled *On the Importance of Pretraining Data Alignment for
Atomic Property Prediction*, introduces a Chemical Similarity Index (CSI)
inspired by FID. CSI measures the alignment between upstream pretraining and
downstream molecular graph distributions. The paper reports that a carefully
selected task-aligned dataset can match or exceed large mixed pretraining at
one twenty-fourth of the pretraining budget, and warns that adding poorly
aligned data can degrade performance.

This is a high-value method-selection result because it challenges the default
assumption that more molecular pretraining data is always better. It is still
not direct MolGap evidence: its upstream sources are the JMP corpora and its
downstream suite is rMD17, QM9, MD22, QMOF, SPICE, and MatBench; the target
contract is atomic-property/energy/force prediction rather than PCQM4Mv2 Gap.

### Public implementation surface

The [official `efficient-atom` repository](https://github.com/Yasir-Ghunaim/efficient-atom)
contains `src`, dataset/structure-mapping scripts, analysis scripts, global
configuration, pretraining commands, fine-tuning commands, and instructions for
extracting upstream/downstream features and computing CSI. It names the same
JMP data sources and exposes individual-dataset versus mixed-pretraining modes,
random versus class-balanced sampling, and checkpoint handoff into fine-tuning.
The repository has only a small visible commit history and follows the JMP
CC-BY-NC licensing boundary; it is a reproducible protocol reference, not a
ready-to-use MolGap dependency.

### Same-database adaptation that remains hypothetical

CSI suggests a useful future *selection* comparison while preserving the
current database: select a train-role subset of PCQM4Mv2 or repaired-2M by a
fixed graph representation, pretrain on that subset, and compare against an
equal-budget random subset and the full same-database pretraining control. The
selection representation, subset size, graph identity, and internal validation
must be frozen before the target result is viewed. This is a protocol idea,
not an authorization. It must not use official validation/test-dev molecules,
target labels, or an external corpus to tune the selector.

**Disposition:** **A/B method and code reference, no current experiment.** The
most important lesson for MolGap is to test alignment versus scale rather than
assuming that a larger external pretraining mixture is automatically useful.

## Cross-paper synthesis

The four papers produce a consistent but bounded conclusion:

1. Pretraining is not a free gain. Strong negative controls are required because
   input features, data balance, and hyperparameters can dominate the objective.
2. Geometry-aware pretraining can be physically meaningful, but force-centric
   objectives need a geometry/label contract that MolGap does not currently
   own. ETKDG-only corruption would be a new, cheaper question.
3. Large cross-domain supervised pretraining is an engineering achievement,
   not evidence that external atomic-property data should replace or augment
   PCQM4Mv2/Track A.
4. Data alignment is a candidate explanation for why a smaller, same-domain
   pretraining pool might be preferable to a larger mismatched corpus. The
   hypothesis is testable only with leakage-safe train-role selection and a
   matched scratch/equal-budget/full-pool control.

## Required evidence before any future admission

Before any pretraining or teacher route from this record becomes a possible
experiment, the protocol must close all of the following:

- exact source revision, checkpoint hash, license, and retrievable artifact;
- PCQM identity overlap and role map, with official validation/test-dev sealed;
- ETKDG construction for both training and inference views;
- a fixed random-init control, a frozen-probe control, and an end-to-end
  fine-tuning control where applicable;
- equal-budget and scale-matched comparisons for any CSI/data-selection claim;
- atomic checkpoints, independently retrievable output chunks, and a mechanical
  acceptance record;
- a separate compute-budget decision before seeds 43/44.

No experiment, database change, checkpoint import, or remote submission follows
from this reading record.

## Primary sources

- [Does GNN Pretraining Help Molecular Representation?](https://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html), [paper PDF](https://papers.nips.cc/paper_files/paper/2022/file/4ec360efb3f52643ac43fda570ec0118-Paper-Conference.pdf), and [supplement](https://papers.nips.cc/paper_files/paper/2022/file/4ec360efb3f52643ac43fda570ec0118-Supplemental-Conference.pdf)
- [ET-OREO / May the Force be with You](https://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html), [paper PDF](https://papers.nips.cc/paper_files/paper/2023/file/e637029c42aa593850eeebf46616444d-Paper-Conference.pdf), and [OpenReview PDF](https://openreview.net/pdf?id=Ge8Mhggq0z)
- [JMP ICLR record](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a896a8bb065774169d9ad65f19208b7-Abstract-Conference.html), [paper PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/4a896a8bb065774169d9ad65f19208b7-Paper-Conference.pdf), and [official code](https://github.com/facebookresearch/JMP)
- [CSI paper](https://arxiv.org/abs/2502.11085), [TMLR/OpenReview record](https://openreview.net/forum?id=jfD9BsrDTb), and [official code](https://github.com/Yasir-Ghunaim/efficient-atom)
