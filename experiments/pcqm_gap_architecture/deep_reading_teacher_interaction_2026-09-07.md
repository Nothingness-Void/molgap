# Deep Reading: Interaction Teachers and New Self-Supervised Algorithms (2026-09-07)

Date: 2026-09-07

This record closes a focused search for methods that combine 2D graphs, 3D
conformers, teacher models, and newer self-supervised objectives. It is an
evidence record, not an experiment protocol. A paper is retained only when a
primary paper and its relevant public artifact or an independently checkable
method result support the stated claim. Search snippets, README headlines, and
unreleased checkpoints are not counted as executable evidence.

MolGap's contract remains unchanged: Track B uses official PCQM4Mv2 roles and
Track A uses the repaired-2M PubChemQC corpus; geometry is ETKDG wherever the
project creates geometry; inference must not depend on a hidden DFT conformer;
and the active architecture screen remains random-initialized. None of the
sources below authorizes a remote run, a checkpoint import, or an external-row
merge.

## 1. IEM: image-enhanced molecular graph representation learning

**Primary sources.** The [IJCAI-2024 paper](https://www.ijcai.org/proceedings/2024/675),
its [PDF](https://www.ijcai.org/proceedings/2024/0675.pdf), and the
[official implementation](https://github.com/HongxinXiang/IEM).

### Research question and data role

IEM asks whether a graph student can inherit information from images rendered
from 2D and 3D molecular views. The paper pretrains an image teacher on about
2M PCQM4Mv2 molecules/conformers. The teacher uses 2D and 3D molecular images,
with ResNet-18 encoders and 512-dimensional representations, and the reported
pretraining run is more than 30 epochs and about 450K steps. This is a
same-database pretraining signal, but it is not a direct PCQM4Mv2 HOMO/LUMO/Gap
experiment.

The five teacher tasks are:

1. image contrastive learning;
2. atom-distribution prediction;
3. bond-distribution prediction;
4. geometry-distribution prediction; and
5. property-distribution prediction.

The priors are therefore not only visual appearance. They encode atom, bond,
geometry, and basic-property information. That distinction matters: copying
the image encoder without the distribution tasks would not reproduce the
reported teacher.

### Distillation mechanism

The graph branch is the deployable student. The teacher is frozen during
downstream training. IEM has two distillation paths:

- a knowledge enhancer that predicts the frozen teacher's pretraining outputs
  and uses an L1 loss on teacher logits/features;
- a task enhancer that matches teacher downstream logits with a smooth-L1 loss
  while also retaining the ground-truth downstream loss.

The paper's graph-only inference path is important for MolGap: images are used
to create the teacher, but the final student can run without images. The result
is still not a free graph-only gain because the graph encoder was trained with
teacher targets derived from a different geometry/rendering contract.

### Evidence and ablations

On the paper's MoleculeNet evaluation, IEM reports an average ROC-AUC increase
from 71.10 to 73.89 for GraphMVP and from 72.51 to 73.81 for Mole-BERT. These
are downstream classification results, not a PCQM Gap number. A view-ablation
is more informative for transfer: using 5% of the rendered images still
reports a gain, while using all images reports a larger gain. This supports the
idea that the teacher's signal can be distilled from a subset, but it does not
show that rasterization is better than an ETKDG graph representation at equal
cost.

### Code and contract audit

The official repository exposes image preprocessing, pretraining, distillation,
model, resume, and asset directories. Its README advertises a 2M-image
pretraining surface and an `IEM.pth` teacher path. The exact downloadable
checkpoint hash, release license, and complete pretraining-data manifest were
not independently closed in this audit; the repository's own TODO section also
indicates that parts of the release surface were still being completed.

The repository's 3D preprocessing path uses RDKit embedding followed by MMFF
optimization. That is not proof of ETKDG-only construction, and the paper does
not supply a train/inference equivalence certificate for MolGap's contract.
The image renderer also adds camera, resolution, PyMOL/RDKit version, and
storage dependencies. No IEM weights or images were downloaded.

**Disposition: B, teacher/distillation reference only.** IEM is useful because
it separates an offline high-information teacher from a graph-only student and
because its frozen-teacher and ground-truth controls are explicit. A future
MolGap adaptation would have to render only ETKDG conformers generated from
permitted train-role molecules, freeze the teacher artifact, prove no external
label leakage, and compare against an identical random-init student. That is a
new protocol, not a current experiment.

## 2. MolInteract: deep 2D--3D interaction rather than late fusion

**Primary sources.** The [PAKDD paper PDF](https://kdd2025.kdd.org/wp-content/uploads/2025/07/paper_22.pdf)
and the [NeurIPS 2024 workshop record](https://neurips.cc/virtual/2024/102819).

### Method

MolInteract uses a 2D GINE tower and a continuous-convolution 3D SchNet tower.
At each interaction layer, the two streams are concatenated, transformed,
split back into modality-specific states, and exchanged again. This is a
materially different question from late prediction fusion: 2D and 3D
information are mixed repeatedly inside the representation.

The cross-modal auxiliary tasks are also explicit. The 2D branch predicts 3D
interatomic distances and bond angles with mean-squared error and predicts
dihedral bins with cross-entropy. The 3D branch predicts 2D bond types,
shortest-path distances, and node eigenvector-centrality ranks with
cross-entropy. The authors describe an eight-layer, roughly 9M-parameter model
trained for 50 epochs on more than 3.3M PCQM4Mv2 molecules.

### Reported result and boundary

The reported downstream table is on QM9, not on the official PCQM4Mv2 Gap
task. The paper gives QM9 frontier errors of 20.60 meV for HOMO and 17.88 meV
for LUMO for its MolInteract row, alongside other QM9 quantities. The paper
also observes that the deep interaction layers matter, even when the
pretraining component is removed in one comparison. This is evidence for an
information-flow mechanism, not evidence for a current MolGap score.

MolInteract's original architecture requires a 3D branch for its interaction
stack. The paper does not establish a graph-only inference mode equivalent to
MolGap's public inference API, and the original 3D source is not demonstrated
to be ETKDG for both training and inference. A targeted search did not locate
an author-maintained code or checkpoint release that closes those gaps. The
paper is therefore stronger as a method reference than as a reusable asset.

**Disposition: B for method, C for implementation.** The transferable idea is
repeated cross-modal interaction with task-specific relation prediction. It
should not reopen the already-tested late SchNet/prediction fusion family. If
ever tested, it needs a 2D-only student control, ETKDG-only 3D inputs, a
separate teacher/representation contract, and an explicit cost comparison.

## 3. LeJEPA: recent predictor-free pretraining and a useful negative lesson

**Primary source.** [LeJEPA: Self-Supervised Pretraining of Molecular Graph
Encoders with LeJEPA](https://arxiv.org/abs/2609.04261), submitted 2026-09-02.

LeJEPA combines a predictor-free joint-embedding predictive architecture with
SIGReg. It is evaluated with both a GPS graph encoder and a Chemprop D-MPNN
control. The paper gives concrete settings for the GPS experiment, including
about 2M parameters, hidden size 128, six layers, four heads, and 128 latent
partitions, plus multi-seed/bootstrap analysis.

The result is not a PCQM or quantum-property result: the reported tasks are
ogbg-molhiv and antibiotic activity. Its most useful evidence for MolGap is a
warning about what counts as a pretraining gain. On ogbg-molhiv, the frozen
linear probe improves substantially over the random representation in the
paper's comparison, while full fine-tuning is only slightly different. The
paper also reports partition sensitivity and complementarity with Morgan
fingerprints. In other words, a pretraining representation can look strong
under a frozen probe without producing a robust end-to-end fine-tuning gain.

The arXiv record states that code, configurations, and checkpoints are
released, but no visible repository link or independently retrievable checkpoint
manifest was closed during this audit. There is therefore no artifact to
import, and no evidence that LeJEPA's result transfers to HOMO/LUMO/Gap.

**Disposition: B/C algorithm observation only.** LeJEPA strengthens the future
pretraining protocol: every MolGap pretraining candidate needs frozen-probe,
fine-tune, scratch, seed, and representation-combination controls. It does not
justify a current PCQM experiment or a new database role.

## 4. Cross-paper synthesis

| Route | What is genuinely new | What blocks direct MolGap use | Safe reusable lesson |
|---|---|---|---|
| IEM | Offline visual teacher with graph-only student inference; teacher and downstream logits are separately distilled | RDKit/MMFF-oriented rendering, incomplete artifact closure, no direct PCQM Gap result | Freeze teacher; report knowledge/task/ground-truth losses separately; keep student inference graph-only |
| MolInteract | Deep repeated 2D--3D interaction with reciprocal relation prediction | QM9 downstream, 3D-dependent original model, no verified code/checkpoint, no ETKDG equivalence | Treat cross-modal interaction as a distinct mechanism from late fusion; require a 2D-only control |
| LeJEPA | Predictor-free joint embedding plus a direct frozen-probe/fine-tune contrast | No PCQM/Gap task and no independently closed public artifact | Frozen-probe gains are not enough; require scratch and fine-tuning controls |

The most defensible future adaptation within the existing database is not to
copy any external weight. It is an ETKDG-only, train-role-only teacher/student
protocol in which the student remains the current GraphState inference path.
The teacher can be an explicitly named auxiliary representation (visual,
relation, or electronic), but the experiment must distinguish:

1. random-init student;
2. teacher-pretrained/frozen student;
3. teacher-pretrained/fine-tuned student; and
4. frozen-probe versus end-to-end transfer.

This decomposition is required before interpreting any gain as pretraining,
distillation, 2D--3D interaction, or target leakage. It is not an
authorization to schedule the run.

## 5. Admission checklist for a future experiment

Before any of these routes can enter a protocol, the following evidence must be
closed in the repository:

- exact paper, code revision, checkpoint hash, license, and artifact manifest;
- canonical identity overlap against PCQM4Mv2 and repaired-2M PubChemQC;
- a role manifest proving that external or teacher labels do not enter the
  supervised target rows;
- ETKDG construction for every conformer used in both training and inference;
- student inference that uses only the public MolGap input contract;
- a fresh random-init control with the same split, parameter budget, seed policy,
  and stopping rule;
- frozen-probe, fine-tuning, and no-teacher controls where pretraining is used;
- atomic checkpoints, per-stage logs, and independently retrievable outputs.

Until those checks pass, the three routes remain literature/engineering
reserve only. The PCQM4Mv2 and repaired-2M databases, current ETKDG rule, active
K3b question, and no-remote-run boundary are unchanged.

## Primary sources

- [IEM IJCAI paper page](https://www.ijcai.org/proceedings/2024/675), [PDF](https://www.ijcai.org/proceedings/2024/0675.pdf), and [official code](https://github.com/HongxinXiang/IEM)
- [MolInteract PAKDD paper PDF](https://kdd2025.kdd.org/wp-content/uploads/2025/07/paper_22.pdf) and [NeurIPS workshop record](https://neurips.cc/virtual/2024/102819)
- [LeJEPA arXiv record](https://arxiv.org/abs/2609.04261)
