# Deep reading: student--teacher, foundation, and structural-pretraining methods

**Date:** 2026-09-07  
**Scope:** molecular foundation models, teacher distillation, learned positional/structural encodings, and completed PCQM/Gap benchmarking assets.  
**Purpose:** expand the evidence library without changing the active database, the ETKDG contract, the frozen architecture decision, or the experiment queue.

## 0. Evidence rule

This record separates four kinds of evidence:

- **Direct target evidence:** the source reports HOMO/LUMO/Gap on a target and
  exposes enough split, geometry, and model information for a fair comparison.
- **Method evidence:** the source has a paper and/or public implementation, but
  its task, geometry, theory level, or data role differs from MolGap.
- **Engineering evidence:** a public repository provides reproducible packaging,
  benchmark, checkpoint, or artifact patterns, but does not justify a new
  scientific candidate.
- **Observation only:** the source is incomplete, closed, or not auditable enough
  to admit as a possible experiment.

The current project target remains the B3LYP/6-31G* Kohn--Sham HOMO/LUMO/Gap
contract on the existing PCQM/Track-A roles.  External pretraining corpora,
weights, QM9/Enamine/UniChem labels, and structural encodings therefore remain
teacher, OOD, or protocol references until a separate contract proves identity,
geometry, and leakage safety.

## 1. GLACIER — a concrete multi-teacher distillation template

**Primary sources:** [paper](https://arxiv.org/html/2606.11382), [official
repository](https://github.com/eemokey/glacier), and the repository's linked
[Hugging Face model](https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol).

### What the paper actually does

GLACIER is a 2026 KDD paper accepted for publication, not a PCQM/HOMO--LUMO
benchmark.  It pretrains three lightweight students on 100,000 molecules sampled
from Enamine REAL:

1. a three-step MPNN with a 300-dimensional graph readout;
2. a two-layer SMILES Transformer with hidden size 128, eight heads, an 8,000
   token BPE vocabulary, and maximum length 512; and
3. an MLP over 217 RDKit physicochemical descriptors.

The student representations are projected into a common space and fused with an
asymmetric Randers/Finsler-inspired gate.  The text embedding supplies a drift
direction, while graph and descriptor embeddings are treated as complementary
keys.  The final fused student is distilled toward fixed embeddings from MiniMol
and MolFormer.  Each teacher has its own projection into a 512-dimensional shared
space.  Teacher weights are predicted dynamically from the student embedding, with
a contribution floor of `epsilon=0.1` and a `-log(tau)` anti-collapse term in the
multi-teacher InfoNCE objective.

The teacher embeddings are extracted once and reused.  The paper reports 250
pretraining epochs in 5.67 hours on one RTX 4080.  Its downstream evaluation is
TDC/MoleculeNet classification and ESOL/LIPO regression; it does not report
PCQM4Mv2 or a HOMO/LUMO/Gap result.  The paper's three-run average AUROC is
`0.799` for the MiniMol-teacher variant, `0.780` for MolFormer, and `0.792` for
the combined-teacher variant.  Those numbers are not MolGap evidence.

The repository is MIT-licensed, pins Python 3.11/PyTorch 2.5.1/CUDA 12.1 in its
README, and exposes a loadable `GLACIER-100k-MiniMol` model.  This is useful
engineering evidence: the model, code, environment, and teacher-embedding
interface are all explicit enough to inspect without inventing a checkpoint.

### What is transferable to MolGap

- Precompute a frozen teacher embedding once and store its hash, model revision,
  input canonicalization, and row identity beside the embedding shard.
- Compare one teacher, the other teacher, and a no-teacher control; do not infer
  that more teachers are better from a multi-teacher headline.
- Use a bounded teacher contribution mechanism only if per-teacher ablations show
  complementary information.  The contribution floor is a useful anti-collapse
  idea, not evidence that the floor is optimal for Gap.
- Keep teacher loss separate from the supervised Gap loss, and measure latency and
  storage because a teacher route can improve representations while degrading
  deployment.

### Contract decision

**Evidence level: B for distillation engineering; C for a current MolGap
candidate.** GLACIER is a teacher-design reference only.  Enamine data, MiniMol,
MolFormer, RDKit descriptors, and TDC/MoleculeNet labels cannot be silently
introduced into the existing PCQM training role.  If a teacher route is opened
after architecture selection, the first admissible design is frozen readout or
auxiliary representation transfer on the existing database, with an identity and
teacher-overlap audit.  No GLACIER experiment is authorized by this reading.

## 2. ChemBERTa-3 — public foundation-model engineering, not Gap evidence

**Primary sources:** [official paper page](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00348b),
[code](https://github.com/deepforestsci/chemberta3), and [Zenodo
release](https://zenodo.org/records/18235841).

ChemBERTa-3 is an open training and benchmarking framework for chemical
foundation models.  The public repository gives a useful reproducibility packet:
environment instructions, pretraining and fine-tuning entry points, benchmark
scripts, explicit featurizers, and a MIT license.  It compares ordinary GCN/RF/
D-MPNN baselines with InfoGraph, InfoMax3D, GROVER, ChemBERTa, and MoLFormer.

The README records scale variants rather than a single universally valid model:
MoLFormer-1.1B uses 100% ZINC20 plus 100% PubChem; the 550M model uses a 50/50
mixture; a smaller MoLFormer model uses a 10% ZINC subset; and the framework also
contains smaller ChemBERTa and 3D/self-supervised configurations.  The featurizer
list explicitly includes an RDKit conformer path for InfoMax3D.  This is useful
for auditing whether a claimed 3D pretraining result actually has a conformer
generator in the input path.

No directly checkable PCQM4Mv2 HOMO--LUMO/Gap result was established from the
audited paper/repository surfaces.  The public framework should therefore not be
treated as a target-matched pretrained checkpoint for MolGap.  Its strongest
borrowable elements are versioned benchmark scripts, data-preparation manifests,
model-size comparisons, and explicit environment recording.  Its external
ZINC20/PubChem pretraining sources require a new role and overlap audit before any
use.

**Disposition: B engineering reference; no candidate experiment.**

## 3. ChemFM — scaling evidence with an incompatible budget and source role

**Primary sources:** [Communications Chemistry paper](https://www.nature.com/articles/s42004-025-01793-8),
[official code](https://github.com/TheLuoFengLab/ChemFM), [Hugging Face
organization](https://huggingface.co/ChemFM), and [Zenodo
record](https://zenodo.org/records/17450883).

ChemFM is a causal language-model foundation study rather than a 3D electronic
property model.  The paper studies scaling up to a roughly 3B-parameter model,
pretrained on 178M UniChem SMILES, and reports scaling analyses plus improvements
across 34 property benchmarks.  The model family and public artifacts make it a
useful reference for how to report tokenization, pretraining data scale, model
size, and downstream adaptation.

The audited paper and public materials do not provide a directly comparable
PCQM4Mv2 Gap result, nor an ETKDG-compatible 3D input contract.  A 3B causal
SMILES model is also outside the current bounded 12-hour architecture-screen
budget.  The only safe lesson for MolGap is methodological: if a future 2D
pretraining route is considered, record source-corpus identity, parameter scale,
tokenization, and target-overlap controls separately.  Do not import UniChem
labels or assume that a general SMILES scaling curve transfers to B3LYP Gap.

**Disposition: B/C foundation-scaling reference; no current candidate.**

## 4. GPSE — learned structural encodings with an important dataset caveat

**Primary sources:** [ICML paper](https://arxiv.org/html/2307.07107), [official
repository](https://github.com/G-Taxonomy-Workgroup/GPSE), and the publicly linked
[PCQM4Mv2 checkpoint](https://zenodo.org/record/8145095/files/gpse_model_pcqm4mv2_1.0.pt).

### Method

GPSE trains an MPNN to reconstruct a bundle of positional and structural
encodings, rather than hand-selecting one encoding for every downstream graph
task.  The paper covers Laplacian eigenvector/positional signals, electrostatic
potential positional encoding, random-walk structural encoding, heat-kernel
diagonal encoding, cycle-count encoding, and related eigenvalue targets.  The
architecture uses random node features, global information through a virtual node,
and multiple prediction heads.  The training loss combines per-encoding L1 and
cosine terms.

The headline evidence is representation-level: on a 5% MolPCBA split the average
PSE reconstruction `R^2` is reported as `0.979`, rising to `0.9979` at 90%;
the paper also reports transfer across graph distributions.  The paper's dataset
appendix explicitly counts unique molecular graphs and distinguishes pretraining
from downstream subsets.  That distinction matters here: a file named
`gpse_model_pcqm4mv2_1.0.pt` being publicly retrievable does not, by itself,
establish that it is an ETKDG encoder, that it was trained without target
information, or that it yields a direct PCQM Gap gain.

### MolGap interpretation

GPSE is a plausible frozen structural teacher or cheap positional-feature probe,
but it is not a direct HOMO--LUMO/Gap method.  Its structural pretexts can be
useful only if a new audit establishes checkpoint provenance, graph canonicalization,
feature dimensions, and no PCQM label leakage.  The repository uses an old
PyTorch/PyG stack; environment compatibility is a practical risk, not a reason
to rewrite the model before its scientific role is accepted.

**Disposition: B method/teacher reference; no initialization or run.**  CondPSE
below is recorded as a negative control against assuming that a stronger structural
pretext must improve molecular property prediction.

## 5. CondPSE — strong synthetic expressivity, no consistent molecular gain

**Primary source:** [paper and full abstract](https://arxiv.org/abs/2607.25169).

CondPSE replaces a fixed learned PSE bottleneck with a polynomial graph-filter
bank driven by Gaussian node probes, then uses FiLM-like conditional modulation
from local, global, and cross-filter signals.  It is pretrained to reconstruct
structural encodings and graph invariants before being frozen for downstream use.

The source reports a large synthetic discrimination improvement relative to GPSE:
CSL accuracy rises from `42.9%` to `97.3%`, and EXP accuracy from `68.3%` to
`99.9%`.  Crucially, it also reports that on real molecular property prediction
CondPSE is comparable to GPSE, with no consistent ordering in a ZINC backbone
sweep.  The authors explicitly discuss the possible mismatch between structural
pretraining targets and molecular property labels.  No official implementation
was found in the audited source.

This is valuable negative evidence for MolGap.  Better graph distinguishability
on synthetic tasks is not enough to admit a pretraining route for Gap.  A future
structural-teacher protocol would need a same-PCQM frozen-control comparison and
an improvement on the target, not only PSE reconstruction or WL-discrimination.

**Disposition: B methodological negative control; do not enter the experiment
queue.**

## 6. GCPE — an incomplete 2026 PCQM claim

**Primary source:** [accepted-manuscript landing page](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-026-60014-0).

GCPE is described as a graph context-aware positional encoding that combines
spatial, spectral, and subgraph structural signals with a gated context-aware
relative positional encoding.  The publisher page describes experiments on
PCQM4Mv2 and other molecular tasks and uses “state-of-the-art” language, but the
audited page does not expose an exact PCQM metric, a complete split description,
an official repository, or an accessible implementation/checkpoint.

**Disposition: observation only.**  The headline claim is not admissible evidence
for a MolGap candidate until the exact table, split, input geometry, code, and
inference information are independently available.

## 7. Chemprop benchmark v2 — a completed reproducibility asset

**Primary source:** [MIT-licensed benchmark repository](https://github.com/chemprop/chemprop_benchmark_v2)
and its linked [Zenodo data archive](https://zenodo.org/records/10078142).

This repository packages Chemprop v2.0.3 benchmark scripts and data, including
both `qm9_gap` and `pcqm4mv2`.  It converts the released per-split CSV files into a
single `data.csv` plus a `splits.json`, so the split is explicit and can be hashed.
The benchmark scripts produce checkpoints and test predictions, and the repository
records the environment in `environment.yml` under an MIT license.

This is a useful completed public baseline/engineering reference, not a new
architecture claim.  It can later serve as an independent sanity check for data
loading, split accounting, metric naming, and artifact retention while keeping
the current database unchanged.  It must not be used to replace the project's
frozen GraphState comparator or to introduce a second PCQM split without a new
protocol.

**Disposition: B engineering/sanity asset; no run authorized.**

## 8. Embedding KD scalability — a closer teacher ablation

**Primary sources:** [Advanced Science paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202503271)
and [official implementation](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties).

This 2025 open-access study is more directly relevant to regression distillation
than a generic foundation-model paper, but its target evidence is QM9 rather than
PCQM4Mv2.  It trains SchNet, DimeNet++, and TensorNet teachers on five QM9
quantum properties (including HOMO, LUMO, and Gap), then distills latent
embeddings into smaller students that predict ten other QM9 properties.  A
separate cross-domain setup transfers a QM9 teacher to ESOL and FreeSolv.

The paper reports architecture-dependent behavior: DimeNet++ students show large
relative `R^2` gains on some properties, while TensorNet often changes little
because its baseline representation is already strong.  The headline “up to
90% `R^2` improvement” is a relative metric and is not a MAE comparison to the
MolGap GraphState anchor.  The source also makes an important negative point:
smaller students can benefit more than larger students, and the benefit varies
by property and architecture.

The implementation is unusually inspectable for a small public project.  It has
separate teacher/student scripts, an L1-plus-cosine latent loss, an
uncertainty-weighted loss option, Optuna tuning, early stopping, checkpointing,
metric logs, and downloadable teacher/student models.  It uses QM9/ESOL/FreeSolv
and a Python 3.9 environment; it is not a PCQM/ETKDG implementation.

### MolGap transfer

The strongest transferable idea is the control matrix: teacher versus no-teacher,
several student capacities, and an embedding-alignment diagnostic.  It supports
testing a teacher without assuming that a larger teacher or a larger student is
better.  However, using a QM9 Gap teacher on the current PCQM Gap would be a
target-proxy route with possible identity and theory mismatch.  A legal future
version would need a teacher trained only on a named, non-overlapping role, frozen
row mapping, and the same ETKDG input path for both teacher and student.  The
paper's QM9 numbers cannot be entered as PCQM evidence.

**Disposition: B, stronger teacher-loss and ablation reference; no current
experiment.**

## 9. EDG — electron-density teacher distilled into a geometry student

**Primary sources:** the [IJCAI-25 paper](https://www.ijcai.org/proceedings/2025/0872.pdf),
the [official MIT-licensed repository](https://github.com/HongxinXiang/EDG), and
the related [EDBench repository](https://github.com/HongxinXiang/EDBench).

### What the paper actually does

EDG is a complete three-stage electronic-teacher pipeline rather than a direct
HOMO/LUMO/Gap predictor pretraining claim:

1. It selects the first 2 million unlabeled conformations and DFT electron
   densities from EDBench, which is built from the PCQM4Mv2 standard. The paper
   states that Psi4 generates the density/ESP grids with grid spacing `0.4`,
   B3LYP, and `6-31G**/+G**`-style basis notation. The density and ESP are
   rendered as six `224 x 224` RGB-D views.
2. ImageED is a ViT-Base/16 masked autoencoder. It trains both masked-token
   prediction and restoration of the hidden patches, with the two Euclidean
   patch losses weighted equally. The released repository exposes an ImageED
   checkpoint and a 2-million-row feature file.
3. A ResNet18 teacher receives four-view structural images, while an MLP maps
   the structural representation into the 768-dimensional ED-feature space
   produced by the frozen ImageED encoder. The teacher is trained with an L1
   alignment loss on the 2-million-molecule pool. The paper reports a 2% held-out
   validation slice, learning rate `5e-3`, batch size `128`, and roughly `280k`
   steps.
4. During downstream distillation, the ED-aware teacher and ED predictor are
   frozen. A geometry student is mapped into the teacher's structural space and
   is trained with Smooth-L1 feature alignment plus the task L1 loss. Images are
   removed at inference; only the geometry student and task head remain.

The public code confirms the role separation: it ships the ImageED checkpoint,
the `200w_ED_feats.pkl` artifact, the ED-aware-teacher checkpoint, and downstream
`teacher_features.npz` inputs. The QM9 command explicitly supports `homo`,
`lumo`, and `gap`, uses a `customized_01` split with seed 42, and takes
`pretrained_pth` and `img_feat_path` as separate inputs. The implementation is
not modern MolGap infrastructure: it pins CUDA 11.6, PyTorch 1.13.1,
PyG 1.6.0, DGL, and Python 3.9, and the large artifacts are linked through
OneDrive.

### What is directly evidenced

The paper reports QM9 experiments with 110K/10K/11K train/validation/test
examples and 12 quantum properties. EDG improves average MAE by 2.2--6.4%
across the four listed geometry backbones. The table includes the relevant
HOMO, LUMO, and Gap columns: for example, EGNN changes HOMO from `29.865` to
`28.319` meV and Gap from `24.696` to `24.283` meV; SphereNet changes HOMO
from `22.007` to `21.842` meV and Gap from `19.435` to `19.014` meV. These are
QM9 results, not PCQM4Mv2 MolGap results. The additional rMD17 energy/force
gains (8.1--33.7% for energy and 1.5--5.3% for force) validate the force-field
use case, but do not establish a frontier-orbital transfer gain on the current
database.

### MolGap contract audit

EDG is the strongest new evidence for an electronic representation teacher, but
it is not admitted as a current experiment for four independent reasons:

- **Potential row overlap:** the teacher pool is explicitly the first 2M
  PCQM4Mv2-derived conformations. The paper does not establish that MolGap's
  official validation/test rows are absent from that pool. The released feature
  index and a row-level identity audit would be mandatory.
- **Theory mismatch:** the ED teacher is built from the paper's
  `6-31G**/+G**` density setup, while MolGap's target contract is B3LYP/6-31G*
  Kohn--Sham HOMO/LUMO/Gap. An ED representation is not automatically a
  same-theory target.
- **Geometry mismatch:** the source teacher pool uses PCQM4Mv2 conformations,
  while MolGap's student contract requires ETKDG for both training and
  inference. Reusing the released teacher features without rerendering or
  proving geometry invariance would break the input contract.
- **Deployment role:** EDG distills a learned ED representation, not a sealed
  target-label teacher. That is scientifically safer than importing Gap labels,
  but it still requires explicit proof that the teacher did not see MolGap
  validation/test identities and that the student can be trained on the exact
  ETKDG representation used at inference.

**Disposition: B, high-value electronic-teacher and artifact-contract reference;
not a current candidate.** Borrow the three-stage role separation, frozen
teacher/mapper pattern, and artifact manifest. Do not import EDBench rows,
pretrained weights, or teacher features into the current architecture without a
new protocol covering identity, theory, geometry, license, and leakage.

## 10. ECMMR and other incomplete hits

The 2026 [ECMMR paper landing page](https://www.sciencedirect.com/science/article/pii/S0957417426009103)
describes BRICS-fragment hypergraphs, node-level contrastive learning, generative
cross-modal latent tasks, PCQM4Mv2 pretraining, and evaluation over 22 downstream
tasks.  The audited public surface did not provide the exact Gap table, split,
code, or checkpoint.  It is therefore an index hit, not a deep-comparable method
or possible experiment.

Likewise, a repository advertising spectral-temporal curriculum for molecular
gaps reports a weak, self-described PCQM test number but does not provide the
auditable protocol needed to compare it with MolGap.  It remains a negative code
lead rather than evidence.  README-only claims are intentionally not promoted by
the existence of a plausible model name or a numerical headline.

## 11. Cross-source conclusions

1. **The new teacher evidence is strongest as engineering, not as a direct Gap
   result.** EDG supplies the clearest electronic-density teacher and frozen
   artifact contract; GLACIER supplies an unusually clear frozen-embedding and
   dynamic multi-teacher contract; ChemBERTa-3 and Chemprop provide reproducible
   artifact packaging; GPSE supplies a structural-teacher design. None supplies
   a matched ETKDG/PCQM4Mv2 Gap improvement.
2. **Structural pretraining needs a target-aligned gate.** GPSE's broad transfer
   and CondPSE's synthetic gains do not justify importing a learned PSE into the
   current model. A MolGap route must beat a same-database random-init control and
   pass target-identity and no-leakage checks.
3. **Teacher embeddings should be treated as data artifacts.** The minimum record
   is model revision, input canonicalization, row identity, embedding shape/dtype,
   hash, and whether the teacher saw any current target labels or validation/test
   molecules.
4. **The database remains unchanged.** Enamine, ZINC20, PubChem, UniChem, TDC,
   MoleculeNet, MolPCBA, and external structural-PSE targets are not merged,
   appended, or used to initialize the current architecture.

## 12. Admission gate if this family is revisited

Before any student--teacher, foundation, or learned-PSE experiment can be
proposed, the record must contain:

1. a frozen teacher/checkpoint hash and license;
2. exact canonicalization and row-level identity mapping to the existing database;
3. proof that official validation/test-dev data and target labels were not used in
   teacher training or selection;
4. an explicit ETKDG-compatible input path, or a separately authorized geometry
   contract;
5. a fresh same-contract random-init GraphState control;
6. one causal hypothesis, one seed-42 screen, and an explicit compute budget;
7. independently retrievable embeddings/checkpoints and a mechanical acceptance
   record.

No source in this file currently clears all seven gates.  This file therefore
expands the literature and asset reserve only; it does not authorize a remote
job, external data merge, pretrained initialization, or production change.

## Primary-source index

- [GLACIER paper](https://arxiv.org/html/2606.11382), [code](https://github.com/eemokey/glacier), [checkpoint](https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol)
- [ChemBERTa-3 paper](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00348b), [code](https://github.com/deepforestsci/chemberta3), [Zenodo](https://zenodo.org/records/18235841)
- [ChemFM paper](https://www.nature.com/articles/s42004-025-01793-8), [code](https://github.com/TheLuoFengLab/ChemFM), [Zenodo](https://zenodo.org/records/17450883)
- [GPSE paper](https://arxiv.org/html/2307.07107), [code](https://github.com/G-Taxonomy-Workgroup/GPSE), [checkpoint](https://zenodo.org/record/8145095/files/gpse_model_pcqm4mv2_1.0.pt)
- [CondPSE](https://arxiv.org/abs/2607.25169)
- [GCPE publisher page](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-026-60014-0)
- [Chemprop benchmark v2](https://github.com/chemprop/chemprop_benchmark_v2), [Zenodo data](https://zenodo.org/records/10078142)
- [EDG paper](https://www.ijcai.org/proceedings/2025/0872.pdf), [code and checkpoints](https://github.com/HongxinXiang/EDG), [EDBench](https://github.com/HongxinXiang/EDBench)
- [ECMMR publisher page](https://www.sciencedirect.com/science/article/pii/S0957417426009103)
