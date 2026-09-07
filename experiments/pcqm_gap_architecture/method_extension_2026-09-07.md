# Method Extension Audit: Pretraining, Teachers, Delta Learning, and New Algorithms

Date: 2026-09-07

This record extends the literature and public-asset audits with a method-level
screen focused on pretraining, teacher/student transfer, delta learning, and
recent graph algorithms. It is evidence only. It does not authorize a remote
job, a dataset merge, a checkpoint download, a pretrained initialization, or a
change to the active Track B queue.

The database decision is explicit: Track B remains the official PCQM4Mv2
dataset and Track A remains the repaired-2M PubChemQC corpus. External datasets
and public models may be used only in a separately named teacher, audit, OOD,
or auxiliary-label role after an identity and license audit. They are not
silent training augmentation and do not replace the main database.

## Executive conclusion

The evidence changes the search priority from another randomly initialized
message-passing block to four method families:

1. **ETKDG-compatible self-supervised pretraining.** Self-Conditioned
   Denoising (SCD) is the strongest newly found implementation-backed lead:
   it has a primary 2026 paper, official code, and a PCQM4Mv2 checkpoint. Its
   published downstream numbers are on QM9 rather than direct PCQM Gap, and
   its geometry pipeline is not yet the MolGap ETKDG contract.
2. **A geometry teacher distilled into the deployed representation.**
   Denoise-and-distill (D&D) is the clearest primary evidence for a frozen 3D
   teacher and graph/node-level student alignment. It uses PCQM4Mv2 and does
   not use PCQM Gap labels during pretraining, but it uses DFT conformers and
   does not provide an official implementation. It therefore needs an explicit
   geometry contract before it can become a MolGap experiment.
3. **Adaptive-noise pretraining.** DenoiseVAE has the strongest new direct-PCQM
   paper number in this pass: its ICLR 2025 appendix reports `0.0777 +/-
   0.0005` on PCQM4Mv2 with `1.44M` parameters. The code audit now shows that
   the public two-commit repository is not a runnable MolGap artifact: the
   default pretraining config points to GEOM, the PCQM builder uses RDKit
   `EmbedMolecule` plus MMFF optimization rather than ETKDG, no checkpoint or
   clear license is exposed, and the public model returns an undefined
   `noneed` variable. The paper/method remains B-level evidence; the code is
   C-level reproducibility evidence, so this is an audit lead rather than a
   permitted initialization.
4. **Same-PCQM low-fidelity delta learning.** DelFTa supplies strong
   open-source evidence that a cheap quantum baseline plus a learned correction
   can outperform direct learning for many orbital endpoints. Its data and
   target level are QMugs and omegaB97X-D/def2-SVP, not PCQM B3LYP/6-31G*.
   A legal MolGap adaptation would compute a low-fidelity proxy on the existing
   PCQM molecules, using ETKDG coordinates at both train and inference, and
   would first pass a CPU residual-variance gate.

The strongest recent exact-PCQM algorithm claim is DeMol (`0.0603 eV` on the
reported validation set), but the official author link currently resolves to
no usable repository or checkpoint. Its pretraining also includes the target
Gap prediction objective, so it is not an architecture-only comparison. It is
retained as a research lead, not an experiment candidate.

MolSpectra strengthens the case for an electronic teacher rather than another
generic graph block: its QM9 ablation reports `26.8 meV` Gap after spectral
pretraining versus `31.8 meV` for coordinate denoising. Its QM9Spectra source
uses B3LYP/def-TZVP and is therefore an external-theory teacher reference, not
a PCQM target or database extension.

No source found in this pass satisfies all of the following simultaneously:
public code or weights, direct PCQM Gap evidence under a comparable role,
ETKDG-compatible train/inference geometry, and a bounded implementation that
is distinct from the accepted GraphState anchor. The active random-init
screen therefore remains unchanged.

## Evidence levels

The labels below are deliberately stricter than “a paper mentions PCQM”.

| Level | Meaning for MolGap | May enter a protocol? |
|---|---|---|
| A | Primary paper plus public implementation or checkpoint, with a clear data and target role. | Only after geometry, split, license, and budget gates pass. |
| B | Primary method evidence or public code exists, but the target, coordinate source, split, or downstream task differs. | Research lead or separately contracted teacher/OOD route. |
| C | Partial code, weak benchmark provenance, dead release, or an unverified headline claim. | No compute allocation; retain for monitoring only. |

An A-level source is not automatically a valid MolGap candidate. It still has
to pass the project-specific ETKDG, official-role, seed-budget, and durable
artifact requirements.

## 1. Pretraining

### 1.1 Self-Conditioned Denoising (SCD) — strongest new pretraining lead

**Evidence.** The primary paper is [Self-Conditioned Denoising for Atomistic
Representation Learning](https://arxiv.org/html/2603.17196v1). The authors
release an [official implementation](https://github.com/TyJPerez/SelfConditionedDenoisingAtoms)
and identify the `ct-scd-pcq` [PCQM4Mv2 pretrained checkpoint](https://huggingface.co/Ty-Perez/ct-scd-pcq).
The repository documents a runnable `pretrain_pcq.yaml` path and states that
the PCQ model is available through Hugging Face; the checkpoint has not been
downloaded or executed by this audit.

SCD differs from ordinary node denoising by conditioning the noisy-geometry
denoising pass on a clean-geometry self-embedding. The paper describes two
forward passes during pretraining, then uses one pass for downstream property
prediction. In its controlled PCQ-pretraining comparison, the CT and CGT
backbones report QM9 Gap MAEs of `24.5 meV` and `19.7 meV`, respectively,
with `3.4M` PCQ structures used for pretraining. The paper also reports that a
random 10% PCQ slice captured most of the downstream QM9 gain, but this is not
evidence for a direct PCQM Gap improvement.

**Why it matters.** The method gives a concrete recipe for a representation
that must encode both local geometric sensitivity and a global molecular
embedding. The implementation exposes a useful engineering detail: the
authors freeze element embeddings during pretraining to avoid downstream
instability, use conditional normalization, and record separate corruption
and regularization noise scales.

**Contract mismatch.** The released path is a TorchMD-Net-style 3D model
operating on PCQM equilibrium coordinates. MolGap inference uses ETKDG, and
the repository hard constraint requires the train and inference conformer
method to match. The released checkpoint therefore cannot be warm-started
into MolGap without a new coordinate/architecture contract. Copying its
weights into GraphState would also invalidate the architecture comparison.

**Disposition: B.** Keep SCD as the first pretraining implementation to
inspect in an evidence-only smoke test. A possible future experiment is an
SCD-inspired denoising head attached to the existing geometry-aware GraphState
using only ETKDG coordinates generated from the permitted PCQM train-role
SMILES. It must be compared with an identical random-init control; the
published noise scales must be re-measured against ETKDG coordinate statistics,
not copied blindly.

### 1.2 DenoiseVAE — strongest direct-PCQM validation lead

The ICLR 2025 [DenoiseVAE paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html)
learns atom-specific Gaussian noise distributions with a Noise Generator and
trains a Denoising Module using reconstruction plus KL regularization. Its
Appendix A.7 reports `0.0777 +/- 0.0005` on the PCQM4Mv2 validation Gap task
with `1.44M` parameters, and the authors release a small
[public repository](https://github.com/liuyurou1/DenoiseVAE). This remains the
strongest direct-PCQM paper number found in the continuation, but it is not a
MolGap-comparable result or a runnable initialization.

The fixed code audit found four independent blockers: the default
[`pretrain_denoisevae.yml`](https://raw.githubusercontent.com/liuyurou1/DenoiseVAE/main/config/pretrain_denoisevae.yml)
targets GEOM block files; the
[`get_3d_lmdb.py`](https://raw.githubusercontent.com/liuyurou1/DenoiseVAE/main/datasets/pcqm4mv2/get_3d_lmdb.py)
path calls RDKit `EmbedMolecule` and `MMFFOptimizeMolecule`; the repository
page exposes no checkpoint manifest or clear license; and the public
[`denoise_prednoise.py`](https://raw.githubusercontent.com/liuyurou1/DenoiseVAE/main/denoisevae/models/denoise_prednoise.py)
returns an undefined `noneed` variable. These findings are evidence that the
artifact needs a new pinned implementation, not evidence that its reported
number is wrong.

A legal adaptation would learn molecule-adaptive noise from ETKDG-generated
train-role coordinates and use ETKDG again for downstream inference, with a
fresh random-init GraphState control. The paper's reported `sigma` and KL
settings are starting points for an audit, not values to copy without
measuring the MolGap coordinate distribution.

**Disposition: B for paper/method, C for code.** Keep DenoiseVAE in the
evidence-only queue. Do not use its PCQM number as a leaderboard result, do not
repair the code silently inside a run, and do not paste its representation
into the architecture screen.

### 1.3 Standard denoising and Fractional Denoising — reproducible references

The [official Pre-training via Denoising repository](https://github.com/shehzaidi/pre-training-via-denoising)
contains a PCQM4Mv2 checkpoint and a documented pretraining command. It is a
useful minimal baseline for noise-prediction loss, but its reported downstream
HOMO/LUMO results are on QM9 and its model expects equilibrium 3D coordinates.

[FradNMI](https://github.com/fengshikun/FradNMI) provides public code and
[Zenodo model files](https://zenodo.org/records/12697467). It extends denoising
with fractional/torsional noise and documents PCQM4Mv2 pretraining. The
released stack is tied to older PyTorch/PyG versions and its coordinate
generation and fine-tuning paths are geometry-dependent. It is a useful
ablation reference for noise design, not a direct MolGap checkpoint. The
paper/repository evidence does not justify reopening the already closed
torsion-state architecture route.

Other public pretraining projects remain useful as design references:

| Source | Borrowable idea | Why it is not a direct candidate |
|---|---|---|
| [UnifiedMolPretrain](https://github.com/teslacool/UnifiedMolPretrain) | Joint 2D/3D masking and position-residual reconstruction. | Older environment and DFT/3D coordinate contract; no matched MolGap GraphState result. |
| [MoleculeSDE / GraphMVPv2](https://github.com/chao1224/MoleculeSDE) | 2D-to-3D and 3D-to-2D diffusion/contrastive pretraining. | Multi-modal geometry route; not ETKDG-compatible by default and not a direct PCQM Gap candidate. |
| [FlexMol](https://github.com/tewiSong/FlexMol) | Two-stage paired/unpaired 2D/3D representation learning. | Requires paired 3D pretraining and has no direct PCQM Gap result establishing a bounded gain. |
| [VideoMol](https://github.com/HongxinXiang/VideoMol) | Multi-frame conformer trajectory as a pretraining view. | Heavy geometry/video pipeline and DFT-conformer assumptions; unsuitable for the 12-hour screen. |
| [3D-GSRD](https://arxiv.org/abs/2510.16780) and [code](https://github.com/WuChang0124/3D-GSRD) | Selective re-mask decoding that prevents a 2D shortcut from making the 3D encoder irrelevant. | No direct PCQM Gap result and no ETKDG-compatible downstream contract. |
| [3D-MolT5](https://arxiv.org/html/2406.05797) and [code](https://github.com/QizhiPei/3D-MolT5) | Discrete local 3D tokens aligned with SELFIES and masked/translation objectives. | PubChemQC/text setting and external data; not an official PCQM Gap result. |
| [MolSpectra](https://arxiv.org/html/2502.16284) and [code](https://github.com/AzureLeon1/MolSpectra) | Spectral patch reconstruction and contrastive alignment as an electronic teacher. | QM9Spectra uses B3LYP/def-TZVP and the source code has no clear license/checkpoint record. |
| [3D-PGT](https://arxiv.org/html/2306.07812) and [code](https://github.com/LARS-research/3D-PGT) | Automated fusion of bond-length, angle, and dihedral pretexts; paper reports PCQM validation `0.0762`. | 42.6M GPS model, DFT pretraining geometry, old stack, and no ETKDG-compatible MolGap comparison. |
| [AniDS](https://arxiv.org/html/2510.22123) and [code](https://github.com/ZeroKnighting/AniDS) | Atom-wise full-covariance anisotropic noise with a public PCQM pretraining path. | 129M force-field model and DFT geometry; no direct PCQM Gap result. |
| [3D-EMGP](https://ojs.aaai.org/index.php/AAAI/article/view/25978) and [code](https://github.com/jiaor17/3D-EMGP) | Equivariant force denoising plus graph-level noise-scale prediction. | GEOM-QM9/MD17, old environment, and no direct PCQM Gap evidence. |
| [Mol-MFFGE](https://www.sciencedirect.com/science/article/pii/S0031320325001918) and [code](https://github.com/Yufei-Luo/Mol-MFFGE) | Task-aware learnable noise transformation and bi-level weighting of denoising plus downstream losses. | GEOM/SPICE/QM9/MD17 rather than PCQM/ETKDG; no direct PCQM Gap result and no independently verified checkpoint. |
| [OCNet](https://www.nature.com/articles/s41524-025-01788-y), [code](https://github.com/545487677/OCNet), and [Zenodo assets](https://zenodo.org/records/14934728) | Conjugated-domain SE(3) pretraining, TB electronic descriptors, public LMDB/checkpoints, and OCELOT H-L Gap evaluation. | OCELOT/TB/DFT/xTB/dimer roles and non-ETKDG geometry; no PCQM-compatible score. |
| [LUMIA](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00713), [MIT code](https://github.com/YajingSun-Group/LUMIA), and [Zenodo](https://zenodo.org/records/15852302) | Chemistry-informed edge/substituent masking, contrastive RGCN pretraining on ~1.4M organic molecules, OCELOT/optoelectronic fine-tuning, substructure explanations, and MCTS search. | No directly verified PCQM Gap result; downstream targets and pretraining roles are organic-optoelectronic/OCELOT, with DGL/RGCN rather than the current GraphState/ETKDG contract. |
| [DFT-to-experiment transfer](https://www.nature.com/articles/s41524-024-01403-6) | Interpretable XGBoost/Klekota--Roth transfer from DFT to experimental frontier levels. | Experimental target and fingerprint/tree model are not PCQM Kohn--Sham prediction. |
| [Conjugated-polymer D-MPNN pretraining](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075011/) | Published comparison of direct training, monomer-DFT pretraining, and TD-DFT-extrapolated polymer pretraining; the cited data/weights URL is currently unresolved. | External experimental polymer targets, TD-DFT/MMFF94s geometries, and no independently retrievable artifact in this audit. |
| [DFT-feature-assisted optical-gap transfer](https://pubs.rsc.org/en/content/articlehtml/2025/nr/d4nr03702b) and [MIT repository](https://github.com/Liu-Group-UF/Machine-Learning-for-Accurate-Optical-Gap-Prediction-in-Conjugated-Polymers) | Public data/code package; modified-oligomer DFT gap plus ECFP6 is a completed electronic-teacher feature protocol. | Experimental optical target and external oligomer/DFT contract; this is feature-assisted transfer, not strict residual delta-learning. |
| [Frontier-orbital Chemprop transfer](https://pubmed.ncbi.nlm.nih.gov/42268043/) | 2026 published abstract/preview reports GFN2-xTB trimer pretraining transferred to chain/bulk gap, IE, and EA. | No verified code, checkpoint, split, data identity, or target-theory packet; literature design reference only. |
| [OPoly26](https://arxiv.org/pdf/2512.23117), [public records](https://materials.colabfit.org/id/DS_wfekwbgncjd3_0), and [fairchem](https://github.com/facebookresearch/fairchem) | Public multi-million-polymer DFT/MD asset with paper-listed HOMO/gap fields and public data/code surfaces. | Polymer condensed-phase/MD geometry and omegaB97M-V/def2-TZVPD differ from B3LYP/6-31G*/ETKDG; public records disagree on basis description and frontier-field coverage. |
| [PubChemQC-to-conjugated-oligomer transfer](https://pubs.rsc.org/en/content/articlehtml/2025/me/d4me00188e) and [ESI](https://www.rsc.org/suppdata/d4/me/d4me00188e/d4me00188e1.pdf) | Same-source B3LYP/6-31G* pretraining on a target-similarity-filtered PubChemQC-100K pool, frozen SchNet layers, and one added interaction block for CO-610 | External target, no official code/checkpoint, unclear coordinate contract, and possible overlap with Track A/PubChemQC lineage; protocol evidence only. |
| [GFN2-xTB/COCONUT gap workflow](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00186b), [code/data](https://github.com/sthinius87/HL-gaps-pub), and [Zenodo](https://doi.org/10.5281/zenodo.15113790) | Public low-fidelity proxy pipeline: ten-conformer xTB/BFGS calculations, Boltzmann aggregation, descriptors, subgroup errors, and OOD theory comparison. | GFN2-xTB target, RDKit/xTB geometry, natural-product distribution, and no same-label B3LYP/6-31G*/ETKDG result; reference only for a future CPU residual audit. |
| [QMCVNet paper](https://authors.library.caltech.edu/records/2jygg-n1r30) and [ChemRxiv DOI](https://doi.org/10.33774/chemrxiv-2021-11r61) | Same-lineage PubChemQC PM6/B3LYP low-fidelity geometry comparison, 3D voxel CNNs, and explicit rotation-augmentation controls | Discussion preprint, no audited code/checkpoint, PM6/MMFF/PM6-aligned geometry mix, and direct high-level prediction rather than a strict residual target; no ETKDG result |
| [QUED](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00411j), [MIT code/models](https://github.com/lmedranos/QUED), and [Zenodo](https://doi.org/10.5281/zenodo.17106019) | DFTB3+MBD global/MO/atomic electronic descriptor with geometry-only/electronic-only/combined ablations; public CPU workflow and artifacts | QM7-X/PBE0+MBD and ADMET tasks, RDKit/MMFF plus CREST/GFN2-xTB geometry, no PCQM/B3LYP/6-31G*/ETKDG result, and feature concatenation rather than a strict residual |
| [POS-EGNN/OMol25](https://pubs.rsc.org/en/Content/ArticleLanding/2026/EB/D6EB00024J), [IBM code](https://github.com/ibm/materials), and [Hugging Face weights](https://huggingface.co/ibm-research/materials.pos-egnn) | HOMO/LUMO/Gap/site-charge multi-task pretraining on >20M OMol25 electrolyte structures; Huber weighting makes HOMO/LUMO primary and Gap a consistency term | OMol25 `ωB97M-V/def2-TZVPD`, explicit MD solvation/ion-pair geometry, public MPtrj weights target energy/force/stress rather than frontier orbitals, and no PCQM/ETKDG result | Multi-task physical-consistency and public 3D foundation design reference only |
| [AEGCNN-MTL](https://www.nature.com/articles/s41524-025-01917-7) | Correlation-grouped QM9 HOMO/LUMO/Gap multi-task result and within-architecture negative-transfer control | QM9/BDG rather than PCQM, no public code/BDG data, no ETKDG path or direct PCQM result | Task-grouping and auxiliary-target admission reference only |

The common conclusion is that pretraining has credible evidence, but a claim
of “pretraining helps MolGap” still needs a same-split, same-geometry,
random-init control. Pretraining on PCQM data is not automatically a free
architecture improvement.

### 1.4 Same-database 2D pretraining without extra labels

If a future method track is opened while keeping the database unchanged, the
lowest-risk variants are:

- masked atom/bond attributes on the official-train-derived molecules;
- graph-view consistency where the perturbations preserve atom identity and
  valid bond semantics;
- path/hop reconstruction or distance-to-edge-bin prediction from the existing
  2D graph; and
- ETKDG coordinate denoising, provided both the clean and corrupted inputs are
  generated by ETKDG and the downstream inference path uses ETKDG as well.

These are candidate objectives, not accepted experiments. Target Gap labels,
official validation/test-dev rows, or a teacher trained on sealed roles must
not be used to construct the pretraining objective.

## 2. Teacher models and knowledge distillation

### 2.1 Denoise-and-distill (D&D) — strongest teacher evidence

The primary source is the [AAAI-25 paper and PDF](https://ojs.aaai.org/index.php/AAAI/article/view/31986).
Its pipeline is unusually relevant to the user’s teacher-model request:

1. pretrain a 3D conformer denoiser;
2. freeze the 3D encoder;
3. distill graph-level or node-level representations into a 2D student; and
4. discard the 3D teacher before downstream inference.

The paper states that the same PCQM4Mv2 collection is used for denoising and
distillation, that each molecule is paired with a single DFT lowest-energy
conformer, and that the PCQM Gap labels are not used during pretraining. It
defines both D&D-GRAPH (mean-pooled feature alignment) and D&D-NODE
(one-to-one atom feature alignment). Across its MoleculeNet evaluation, D&D
beats the no-pretraining control in 9/10 tasks, with the paper reporting
average improvements of 4.6% for classification and 18.6% for regression.

**Transferable design.** For MolGap, the important idea is not the TokenGT
backbone. It is the role separation: a frozen geometry-aware teacher provides
representation targets, while the deployable student is trained on the
features available at inference. Node-level alignment is especially relevant
because the PCQM graph and conformer share atom identity.

**Hard blocker.** D&D uses DFT coordinates for the teacher. Directly training
the student against DFT-coordinate teacher states and then deploying it with
ETKDG would violate the existing train/inference geometry rule unless the
project deliberately creates and approves a privileged-geometry pretraining
contract. A legal but weaker adaptation would use ETKDG for both teacher and
student; its benefit is an empirical question and is not established by the
D&D paper.

No official D&D code or checkpoint was found in this audit. The paper is strong
method evidence but not a drop-in reproducibility asset.

**Disposition: B.** Retain as the template for a future teacher protocol, not
as an immediate remote candidate.

### 2.2 Earlier PCQM teacher precedents

- [Coordinating Cross-modal Distillation](https://arxiv.org/abs/2211.16712)
  reports 3D-to-2D global and atom-level distillation on PCQM4Mv2 and a
  challenge result, but its DFT coordinate role and unavailable implementation
  prevent direct adoption.
- The [ViSNet PCQM technical report](https://ogb.stanford.edu/paper/neurips2022/pcqm4mv2_ViSNet.pdf)
  documents a pretrained 3D ViSNet that transfers geometric information from
  optimized structures to generated structures and reports a strong ensemble
  challenge result. This is useful precedent for teacher/geometry separation,
  but it is already architecture evidence in the project and not a new
  random-init route.
- [Knowledge Distillation for Molecular Property Prediction](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties)
  provides a public generic implementation with teacher/student SchNet,
  DimeNet++, and TensorNet options, plus feature/cosine and uncertainty-aware
  losses. It is not a PCQM project, and the repository is small; borrow the
  loss and checkpoint structure only after independently checking the code.

The associated open-access [Advanced Science study](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202503271)
is stronger than the repository alone as a teacher ablation reference. It trains
teachers on five QM9 quantum properties, including HOMO, LUMO, and Gap, then
distills latent embeddings into smaller students for ten other QM9 properties;
it also tests QM9-to-ESOL/FreeSolv transfer. The reported relative `R^2` gains
are architecture- and property-dependent, with DimeNet++ more responsive than
TensorNet in the reported setup. The public code exposes an L1-plus-cosine
latent loss, uncertainty weighting, Optuna, checkpointing, and a no-KD switch.
This supports a future control matrix, but it does not establish PCQM/ETKDG
evidence and does not permit a QM9 teacher to be used silently.

The safe teacher target hierarchy is therefore: local electronic or geometric
representation first, teacher property prediction second. Teacher Gap outputs
are dangerous because they can duplicate the target and conceal label leakage;
any such route requires out-of-fold predictions and a role manifest.

### 2.3 Electronic teachers and auxiliary targets

Several recent papers use charges, bond orders, electron density, or other
local quantum descriptors as a teacher signal. They are scientifically
interesting but not direct PCQM Gap evidence:

- [HEDMoL](https://arxiv.org/html/2602.07087) retrieves electron-level
  substructure attributes and learns atom/electron representations. Its
  external QM9 retrieval creates an explicit identity-leakage concern for any
  QM9 evaluation.
- [MET](https://doi.org/10.1039/D5ME00173K) uses charge-supervised equivariant
  pretraining and a transformer adaptation stage. Its evidence supports
  charge supervision as a teacher signal, not an automatic EGNN transplant.
- [Atom-in-a-molecule quantum-property pretraining](https://link.springer.com/article/10.1186/s13321-025-00970-0)
  compares atom-level quantum-property pretraining, Gap pretraining, and atom
  masking. The evidence favors local atomic properties over using the same
  molecular Gap as the sole pretraining target, but its downstream benchmark
  is ADMET rather than PCQM Gap.
- [Q-GEM](https://doi.org/10.1002/advs.202504867) uses geometry and CM5/Wiberg
  electronic self-supervision. Its reported gains are modest and coupled to
  other datasets and coordinate methods, so it is a teacher-pattern reference.
- [Overlap-population prediction](https://doi.org/10.1093/chemle/upaf038)
  supplies a bond-level electronic interaction target. It is based on QM9 and
  PBE/FHI-aims, so it requires a no-overlap audit before any use as a teacher.
- [EDG](https://www.ijcai.org/proceedings/2025/0872.pdf) is the most complete
  electronic-density teacher package found in this search. It pretrains
  ImageED on 2M EDBench/PCQM-derived DFT density views, trains a structural
  ResNet18 teacher to predict the frozen ED representation, and distills that
  representation into a geometry student with Smooth-L1 alignment. Its QM9
  table includes HOMO/LUMO/Gap, and its [official repository](https://github.com/HongxinXiang/EDG)
  exposes checkpoints and feature artifacts. It is still only a B-level
  reference: the density theory, source conformers, PCQM row overlap, and old
  software stack do not match the current B3LYP/6-31G*/ETKDG contract.

These sources support a future hypothesis: if a teacher route is opened, local
electronic descriptors should be tested before a large target-prediction
teacher. The labels, level of theory, geometry, and license must be recorded
per source; “charge” is not a method-independent target.

### 2.4 GLACIER and learned structural encoders

[GLACIER](https://arxiv.org/html/2606.11382) adds a useful implementation-backed
variant of the teacher route. It precomputes fixed MiniMol/MolFormer embeddings,
projects each teacher independently, and learns a bounded dynamic contribution
for each teacher through a multi-teacher InfoNCE loss. Its graph/SMILES/RDKit
descriptor fusion is lightweight and its [MIT repository](https://github.com/eemokey/glacier)
provides a loadable model, but all reported downstream tasks are TDC/MoleculeNet;
there is no direct PCQM Gap result. The transferable object is the artifact and
loss contract, not the Enamine corpus or its headline scores.

[GPSE](https://arxiv.org/html/2307.07107) and [CondPSE](https://arxiv.org/abs/2607.25169)
give a useful paired warning. GPSE learns a shared representation of several
positional/structural encodings and has public code plus a PCQM-named checkpoint;
CondPSE improves synthetic structural discrimination but reports no consistent
molecular-property advantage over GPSE. Together they argue that a learned
structural teacher must be judged on a matched Gap control, not on PSE
reconstruction or synthetic WL expressivity alone.

[ChemBERTa-3](https://github.com/deepforestsci/chemberta3) and
[ChemFM](https://www.nature.com/articles/s42004-025-01793-8) are foundation-model
engineering references. Their public configurations, corpus accounting, and
benchmark packaging are useful, but their ZINC/PubChem/UniChem roles, model
scales, and missing direct PCQM Gap evidence exclude them from the current
candidate pool. The complete batch is recorded in
[deep_reading_foundation_teacher_structural_2026-09-07.md](deep_reading_foundation_teacher_structural_2026-09-07.md).

## 3. Delta learning

### 3.1 DelFTa — open-source delta evidence, wrong target contract

[DelFTa](https://github.com/josejimenezluna/delfta) is a maintained public
toolbox with an [API/documentation site](https://delfta.readthedocs.io/en/latest/)
and model releases. It explicitly supports both direct learning and delta
learning with a GFN2-xTB baseline, and predicts HOMO, LUMO, Gap, dipoles,
charges, and bond orders. The associated paper,
[Delta-QML for medicinal chemistry](https://pmc.ncbi.nlm.nih.gov/articles/PMC9093086/),
reports that delta learning outperforms direct learning for most QMugs
endpoints and discusses the correlation between the GFN2-xTB baseline and
the higher-level reference.

The scientific pattern is:

```text
b(x)       = cheap baseline property on the same molecule
r(x)       = model prediction of y(x) - b(x)
final(x)   = b(x) + r(x)
```

For the released DelFTa system, `b(x)` comes from GFN2-xTB and the reference
target is omegaB97X-D/def2-SVP on QMugs. That is not the PCQM B3LYP/6-31G*
target, and DelFTa’s geometry path is not the MolGap ETKDG contract.

**A database-preserving adaptation.** A future MolGap delta screen could
compute a low-fidelity single-point proxy `b(x)` for the existing PCQM
official-train molecules, without importing a new labeled database. The
proxy would need to be generated from the same ETKDG coordinates used by
MolGap inference, with unit and charge/spin conventions frozen. The model
would learn the PCQM Gap residual only on the permitted train split.

This is not yet an experiment because the key prerequisite is unknown: whether
the chosen proxy is sufficiently correlated with PCQM Gap to make the residual
easier than direct learning. A CPU-only audit must first measure, on the
existing official-train-derived internal split:

- proxy coverage and failure rate;
- `MAE(b, y)`, correlation, and residual scale;
- residual scale by molecule size, element set, and conjugation proxy;
- the cost per molecule and projected full-data CPU cost; and
- whether the proxy uses exactly the same ETKDG coordinate convention at train
  and inference.

Only if the residual gate is favorable should a paired seed-42 direct-vs-delta
screen be written. It must use an independently fresh direct GraphState
control, identical optimizer/budget, and an explicit final prediction
definition. The delta route must not receive a same-row in-sample prediction
from a learned baseline. If a learned baseline is ever used instead of xTB,
its train-role outputs must be out-of-fold.

**Disposition: B.** DelFTa is strong evidence for the method and a useful
implementation reference, but it does not license importing QMugs labels or
using its checkpoint for PCQM. The same-PCQM proxy version is a conditional
future question, not a silently approved augmentation.

### 3.2 Delta routes that should not be admitted

- GW-vs-DFT delta learning is scientifically valid, but it predicts a different
  target and cannot improve a B3LYP/6-31G* PCQM target without a new target
  contract. See [Interpretable delta-learning of GW quasiparticle energies](https://doi.org/10.1088/2632-2153/acf545).
- QMugs, QM7-X, QO2Mol, or mixed OPV databases cannot become PCQM labels by
  renaming the target column. Their geometries, functionals, bases, and
  molecular distributions differ.
- An out-of-fold residual from an existing MolGap model is close to the
  already-closed residual/fusion family. It should not be reopened under the
  name “delta learning” without a distinct scientific question and a fresh
  control.

## 4. Recent algorithms and completed codebases

### 4.1 DeMol — important claim, not reproducible evidence yet

[DeMol](https://arxiv.org/html/2603.00568) is an ICLR 2026 paper that models
both atom-centric and bond-centric graphs and couples them with atom--bond and
bond--bond interactions. The paper reports `0.0603 eV` on its PCQM4Mv2
validation comparison with a `186M` parameter single model. Its auxiliary
objectives include masked atom prediction, coordinate recovery, and bond
prediction; notably, the PCQM experiment states that Gap prediction is already
included in pretraining and the model is evaluated without an additional
fine-tuning stage.

This is not comparable to the MolGap random-init architecture screen for two
independent reasons: the model is much larger, and its target label participates
in pretraining. The author-linked [GitHub URL](https://github.com/LiuYunqing/DeMol)
currently does not expose a usable repository or checkpoint. The result is
therefore retained as a lead for future source monitoring only.

### 4.2 DGT — complete pretraining/code/data reference, no direct MolGap admission

[Dual Graph Transformer](https://github.com/zhangsy-ryan/DGT) provides public
code for separate atom and bond graphs with cross-level interactions and
optional 3D descriptors; the related [paper](https://www.nature.com/articles/s41467-026-75005-9)
reports QM9 ablations and a PCQM4Mv2 Gap-pretraining transfer protocol. The
paper uses random `10K`, `100K`, and `1M` PCQM subsets for 100-epoch Gap
pretraining (AdamW, batch 128, learning rate `2e-4`, cosine schedule, five
warmup epochs), then initializes QM9. At 10K, the reported QM9 HOMO/LUMO MAEs
are `0.0306/0.0240 eV`; the paper reports no significant downstream gain from
larger pretraining subsets. The [Zenodo code archive](https://doi.org/10.5281/zenodo.20009509)
and [Figshare source data](https://doi.org/10.6084/m9.figshare.30665129) make
the asset surface stronger than a paper-only lead.

This still does not establish a direct, ETKDG-compatible PCQM Gap gain for the
MolGap contract. The visible repository exposes QM9 configurations rather than
a PCQM pretraining config/checkpoint; the downstream QM9 labels use
B3LYP/6-31G(2df,p), the 3D comparison uses DFT/MMFF/UFF geometry, and the bond
graph is conceptually close to the project’s persistent EdgeState while the
existing GraphState anchor already contains a bond-level state. Reimplementing
DGT wholesale would therefore be mostly a large architecture transplant, not a
clean new information-flow question.

**Disposition: B for complete pretraining/code/data reference, C for direct use,
and not a new queue item.** A specific bond--atom interaction could be proposed
only with an explicit new hypothesis that is not already covered by
EdgeState/GraphState evidence, uses ETKDG for every geometry view, and includes
a no-pretraining control.

### 4.3 Weak or non-comparable public claims

The public repository
[spectral-temporal-curriculum-molecular-gaps](https://github.com/A-SHOJAEI/spectral-temporal-curriculum-molecular-gaps)
claims PCQM training and reports `0.268 eV`, but has zero stars, three visible
commits, no paper or independently traceable run provenance, and a result far
below the established PCQM references. Its README is not sufficient evidence
for a candidate and it is excluded.

The artifact scan also found three useful direct-PCQM engineering references:
GraphGPT exposes source, OGB preprocessing, four ModelScope PCQM checkpoints,
and a graph-to-sequence NTP/SMTP pretraining path; TGT exposes data preparation,
distance-prediction stages, and TGT-At/TGT-Agx2 weights; and Graphcore GPS++
exposes durable IPU-specific training/inference and 11M/22M/44M checkpoint
layouts. These are completed-system references, not current candidates: their
geometry and accelerator contracts are not the MolGap ETKDG/single-model
contract. MooseML is useful only as a deployment pattern (GINE plus six RDKit
descriptors around an OGB PCQM model), while the spectral-curriculum repository
has no primary paper, independent metric, or visible checkpoint and remains a
C-level negative artifact. The detailed source and command audit is in
[`deep_reading_public_code_2026-09-07.md`](deep_reading_public_code_2026-09-07.md).

The same rule applies to any recent paper that reports validation-as-test
numbers, a single unseeded run, an external geometry source, or a stale
leaderboard comparison without a reproducible artifact. Search-result snippets
and repository README claims are discovery signals, not acceptance evidence.

### 4.4 IEM, MolInteract, and LeJEPA: interaction and pretraining controls

The [IEM paper](https://www.ijcai.org/proceedings/2024/675) and
[official code](https://github.com/HongxinXiang/IEM) add a same-PCQM image
teacher with atom/bond/geometry/property distribution tasks and a graph-only
student. The teacher/student separation is worth borrowing, but the visible
3D preprocessing is RDKit/MMFF-oriented, the exact teacher release is not
closed, and the reported downstream evidence is MoleculeNet rather than direct
PCQM Gap.

[MolInteract](https://kdd2025.kdd.org/wp-content/uploads/2025/07/paper_22.pdf)
is a repeated 2D GINE--3D SchNet interaction model. Its reciprocal relation
losses predict distances, angles, dihedrals, bonds, shortest paths, and
centrality, with PCQM pretraining and QM9 frontier evaluation. It is a useful
mechanism reference because it is not late prediction fusion, but no
author-maintained code/checkpoint was verified and the original model is
3D-dependent.

[LeJEPA](https://arxiv.org/abs/2609.04261) is a recent predictor-free
joint-embedding objective with SIGReg. It does not use PCQM or quantum Gap
tasks, but its frozen-probe versus fine-tune evidence is directly relevant to
pretraining evaluation. A MolGap pretraining report must not call a frozen
probe gain an end-to-end gain without a matched scratch/fine-tune control.

**Disposition.** IEM and MolInteract remain future teacher/method references;
LeJEPA remains an algorithm observation. None permits a database change,
external-weight import, or current remote experiment.

### 4.5 Pretraining controls, force objectives, and task-aligned data

The [NeurIPS GNN-pretraining study](https://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html)
is a required negative/control reference. It varies objectives, splits, input
features, pretraining scale, and GNN architectures, and reports that
self-supervised gains can be negligible while supervised gains can shrink with
richer features or balanced splits. It does not use PCQM4Mv2 Gap and no
verified author artifact was closed. The practical consequence is that any
future MolGap pretraining claim needs scratch, frozen-probe, fine-tune, feature,
and optimizer controls.

[ET-OREO](https://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html)
adds a force-centric equilibrium/off-equilibrium objective over more than 15M
conformations, with zero-force regularization, force-guided denoising, and
direct force supervision. It is a strong method reference but uses force labels,
external conformers, and explicit 3D geometry; it has no direct PCQM Gap score
or verified artifact in this audit. Only an ETKDG-only perturbation objective
could be considered later.

[JMP](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a896a8bb065774169d9ad65f19208b7-Abstract-Conference.html)
is a public multi-domain supervised pretraining/configuration package. The
[official repository](https://github.com/facebookresearch/JMP) exposes configs,
dataset instructions, fine-tuning paths, and named checkpoints, but is archived,
does not bundle its large data, and is majority CC-BY-NC. Its OC20/OC22/ANI1x/
Transition-1x energy/force contract is external to MolGap.

[CSI](https://arxiv.org/abs/2502.11085) and the public
[`efficient-atom` repository](https://github.com/Yasir-Ghunaim/efficient-atom)
focus on upstream/downstream graph alignment instead of raw pretraining scale.
The paper reports a 1/24-budget task-aligned selection result and a degradation
when poorly aligned data are added. The code exposes feature extraction, CSI
calculation, selection, pretraining, and fine-tuning scripts, but all evaluated
datasets remain the JMP atomic-property suite. A same-database train-role-only
selector is a future hypothesis, not a current experiment.

**Disposition.** These sources strengthen the method ledger but do not broaden
the database, geometry method, or active random-init architecture screen.

### 4.6 Direct-PCQM attention and positional-encoding audit

The direct-PCQM attention batch is useful because it separates score claims
from transferable mechanisms. [GRPE](https://arxiv.org/html/2201.12787) and
[GPTrans](https://arxiv.org/pdf/2305.11424) provide the strongest completed
2D references: GRPE puts topology and edge context into both attention and
values, while GPTrans explicitly propagates node information to edges and
back. [TokenGT](https://arxiv.org/html/2207.02505) supplies a clean
node/edge-token plus Laplacian-identifier baseline, and
[(2,1)-GT](https://proceedings.mlr.press/v235/muller24c.html) shows how a
connected tuple tokenizer can keep higher-order information at O(n+m) token
count. These are architecture references only; none authorizes a public
checkpoint warm start during the current random-init screen.

[GEM-2](https://arxiv.org/pdf/2208.05863) and
[TetraGT](https://proceedings.iclr.cc/paper_files/paper/2026/hash/239b0f62a2cb86876a0c7028393d2a18-Abstract-Conference.html)
have direct PCQM evidence for higher-order interaction, but both are bounded
by large models and geometry contracts that include RDKit/MMFF-style or other
non-ETKDG coordinates. [Edge Transformer](https://arxiv.org/pdf/2401.10119)
has an attractive 3-WL-style pair-state formulation, but its cubic pressure
and single-seed validation table make it an upper-bound reference.

[GAPE](https://arxiv.org/html/2505.13087) is the most relevant new PE idea:
its graph-alignment pretraining is topology matched and reports a strong
small-subset PCQM comparison. The result is nevertheless an edge-feature-free
Transformer contract with a transductive pretraining question, so a future
test would need frozen/joint/no-PE controls and an explicit train-only versus
all-topology declaration. [GFSA](https://arxiv.org/pdf/2312.04234) is a useful
operator repair but its four-seed PCQM4Mv2 gain is only `0.0002`;
[Specformer](https://arxiv.org/pdf/2303.01028) is retained as a spectral
implementation reference because its validation-as-test protocol is not
admissible. Quantum-walk PE, AdvSynGNN, and Edge-Set Attention remain
negative or low-priority controls for the same reason: single-run,
classification-oriented, or validation-as-test evidence is not enough.

The batch therefore changes prioritization, not the queue. If a future budget
opens after the current causal screen, the smallest defensible probes are
GRPE-style relative value context, a train-role-only GAPE PE, or a compact
node--edge propagation block. Each would need the accepted graph cache, the
same ETKDG roles, a no-pretraining/no-PE or no-propagation control, paired
seed-42 governance, and independent artifact paths.

### 4.7 Fragment and multi-resolution pretraining

The 2026-09-08 fragment continuation is recorded in
[deep_reading_pretraining_fragment_multiscale_2026-09-08.md](deep_reading_pretraining_fragment_multiscale_2026-09-08.md).
MolCHG is the lowest-risk new hypothesis because its atom/bond/fragment/graph
source tasks can be regenerated from the permitted train role without
privileged geometry. BiScale-GTR is a stronger but more failure-prone
learned-token alternative: Graph-BPE determinism, validity filtering, OOV
recursion, and round-trip reconstruction must pass a CPU gate first.

FragNet contributes fragment-connection attribution but its Uni-Mol/RDKit
3D workflow is not the current ETKDG route. MORE contributes the most useful
pretraining-control design, especially frozen probing, fine-tuning, and
leave-one-out source-task ablations, but its ZINC2M and random/MMFF roles
cannot be imported. FragmentNet is a learned-tokenizer idea without a
verified official artifact. All five remain outside the active queue until
K3 is resolved and a separate same-database protocol is accepted.

### 4.8 Teacher distillation and conformer-alignment continuation

The 2026-09-08 teacher/geometry read is recorded in
[deep_reading_teacher_geometry_2026-09-08.md](deep_reading_teacher_geometry_2026-09-08.md).
It adds two different kinds of evidence. PG-MLD makes a dynamic privileged
teacher explicit: an equivariant 3D encoder and Liquid Time-Constant network
summarize perturbed/trajectory frames and formal/Gasteiger channels, then a
frozen teacher transfers atom- and molecule-level signals into a SMILES
student. ConforFormer uses a simpler identity-invariance target: two
conformers of the same molecule are NT-Xent positives on top of Uni-Mol's
masked-token, coordinate, and distance objectives, with frozen-CLS probing
separated from fine-tuning.

Neither paper reports a matched PCQM4Mv2 B3LYP/6-31G* Gap result. PG-MLD's
OpenMM/ZINC trajectory, charge, and language-model roles are not the MolGap
contract, and its public repository omits the large data/checkpoint assets.
ConforFormer's Uni-Mol/MMFF/OMol geometry is not ETKDG, and its checked-in
loss contains unexplained distance-normalization constants. Therefore the
portable questions are deliberately narrow: M11 is an ETKDG conformer-
alignment probe after K3, and M12 is a lower-priority ETKDG/PCQM rewrite of
atom/molecule teacher alignment only if a 2D student route is separately
authorized. M11 and M12 cannot be stacked or used to reopen the active
random-init architecture screen.

## 5. Priority order if a future method budget opens

The following is a research ordering, not authorization. It assumes the active
K3b and already released seed-42 questions have been closed or retained under
their own records.

| Order | Future question | Required first gate | Why it is worth testing |
|---|---|---|---|
| M0 | Evidence-only smoke/hash of SCD code and `ct-scd-pcq` metadata. | Repository revision, checkpoint hash, license, forward-pass smoke, coordinate expectations. | Highest reproducibility among new pretraining leads. |
| M1 | ETKDG-only denoising pretraining of the existing GraphState geometry path. | Same ETKDG construction; no target labels; paired random-init control; CPU memory/throughput estimate. | Preserves the database and tests SCD’s objective without importing its incompatible checkpoint. |
| M2 | ETKDG-compatible teacher/student representation alignment. | Teacher/student role manifest, frozen teacher, node/graph alignment, no sealed labels, exact student inference path. | Transfers the best D&D idea while respecting deployment inputs. |
| M3 | Same-PCQM cheap-proxy delta learning. | CPU proxy coverage, residual variance/correlation, cost projection, unit/charge/spin audit. | Tests whether a physical low-fidelity baseline removes learnable error. |
| M4 | Electronic auxiliary teacher using a named, deduplicated source. | Source license, canonical identity overlap, target-level metadata, teacher-only role, no external-label merge. | Tests whether local electronic information helps more than another graph block. |
| M5 | ETKDG-only image/interaction teacher probe inspired by IEM or MolInteract. | Frozen teacher manifest, graph-only student inference, 2D-only control, exact rendering/3D contract, and paired cost estimate. | Tests whether repeated or visual cross-modal supervision adds information without reopening late fusion. |
| M6 | Pretraining evaluation-control matrix inspired by LeJEPA. | Scratch, frozen-probe, fine-tuned, seed, and representation-combination controls on the same internal split. | Prevents a probe-only pretraining gain from being misread as a production gain. |
| M7 | Equal-budget, train-role-only same-database pretraining data-alignment screen inspired by CSI. | Frozen graph representation, selector fit only on train-role data, random-subset and full-pool controls, and no official validation/test-dev access. | Tests whether alignment rather than raw same-database scale is the limiting pretraining factor. |
| M8 | ETKDG-only clean/perturbed geometry objective inspired by ET-OREO. | Same ETKDG source for both views, no force labels, one objective, paired scratch control, and CPU/GPU cost estimate. | Tests the transferable consistency idea without importing external conformers or force fields. |

| M9 | Same-database MolCHG-style fragment/source-task pretraining. | Train-role-only fragment vocabulary/labels, deterministic cache, scratch/no-auxiliary/frozen-probe controls, equal budget, and K3 closure. | Lowest-risk new multiscale hypothesis because it does not require external geometry or a public checkpoint. |
| M10 | Same-database BiScale-GTR-style Graph-BPE masked-fragment pretraining. | CPU tokenizer validity, OOV recursion, round-trip reconstruction, frozen vocabulary/identity manifest, and equal-budget atom-only control. | Tests context-aware learned fragments after the simpler source-task route, with explicit implementation-risk accounting. |
| M11 | ETKDG conformer-identity alignment inspired by ConforFormer. | Two accepted ETKDG views per train-role identity, pair/negative manifest, one-conformer rate, scratch/no-contrastive/frozen-probe/fine-tune controls, and deterministic deployment path. | Cleanest new geometry-pretraining hypothesis; tests conformer invariance without external trajectories or charge labels. |
| M12 | ETKDG/PCQM atom-and-molecule teacher alignment inspired by PG-MLD. | Explicit 2D-student authorization, train-role-only teacher source tasks, frozen teacher, no MD/charge import, teacher/student hashes, and cost estimate. | Tests privileged geometry transfer after simpler routes; dynamic trajectory/LTC components are not first-line under the current budget. |

M1--M4 and M9--M12 each require a separate dated protocol. They cannot be stacked into one
large screen: the result would not identify whether denoising, distillation,
the proxy, the auxiliary label, data selection, or the perturbation objective
caused a gain.

## 6. Admission checklist

Before any of these routes can become a possible experiment, the candidate
card must contain:

1. primary paper URL, repository URL, exact revision, checkpoint URL, and
   dataset release;
2. retrieval status, file size, checksum, and an independently repeatable
   forward or preprocessing smoke test;
3. license and any non-commercial/share-alike restrictions;
4. target, unit, sign convention, functional, basis, code version, charge,
   spin, convergence, and geometry metadata;
5. canonical identity and conformer deduplication against all relevant MolGap
   roles;
6. a role map proving that official validation/test-dev labels and future
   sealed data do not enter selection;
7. an explicit ETKDG train/inference decision, or a separately authorized new
   geometry contract;
8. parameter, memory, throughput, CPU/GPU, and 12-hour budget estimates;
9. a fresh matched control, one causal hypothesis, and a numerical stop gate;
10. atomic checkpoints, independently retrievable output chunks, and a dated
    mechanical acceptance record.

The minimum scientific comparison for a pretraining, teacher, or delta route
is a fresh same-contract random-init GraphState control on the frozen internal
split. A seed-42 result is a promising signal only; seeds 43/44 require a
separate confirmation-budget decision. No method in this record changes the
official validation/test-dev or full-data gates.

## 7. Final disposition

| Route | Evidence level | Include as possible method? | Reason |
|---|---:|---:|---|
| SCD with public PCQM checkpoint | B | Yes, future pretraining lead after ETKDG adaptation | Primary 2026 paper, official code, public checkpoint; downstream task and geometry differ. |
| DenoiseVAE | B paper / C code | Yes, paper/method audit only; no current initialization | Direct PCQM4Mv2 validation number, but public code points to GEOM, uses RDKit+MMFF for PCQM construction, lacks a visible checkpoint/license, and has an undefined return variable. |
| IEM image teacher | B | Teacher/distillation reference only; no current initialization | Same-PCQM image pretraining and graph-only student inference are useful, but the visible 3D route is RDKit/MMFF-oriented, the artifact manifest is incomplete, and no direct PCQM Gap result is reported. |
| MolInteract | B method / C artifact | Cross-modal method reference only; no current experiment | Repeated 2D GINE--3D SchNet interaction and reciprocal relation tasks are distinct from late fusion, but the paper is QM9-downstream, 3D-dependent, and lacks a verified code/checkpoint release. |
| LeJEPA | B/C | Algorithm and control reference only; no current initialization | Predictor-free joint embedding plus SIGReg supplies a frozen-probe/fine-tune warning, but it has no PCQM Gap task and no independently closed artifact. |
| Does GNN Pretraining Help Molecular Representation? | B/C | Negative/control reference only; no current initialization | Controlled objective/split/feature/scale/architecture ablations warn that self-supervised gains may vanish and hyperparameters/features can dominate; no PCQM Gap or verified artifact. |
| ET-OREO | B/C | Force-centric pretraining reference only; no current initialization | More than 15M-conformation equilibrium/off-equilibrium objective and force/MD gains are credible, but force labels, external 3D conformers, no direct PCQM Gap result, and no verified artifact violate the current contract. |
| JMP | A/B | External multi-domain pretraining/configuration reference; no current initialization | ICLR paper and archived official code expose large multi-task pretraining, configs, and named checkpoints, but data/theory/geometry are external, data are not bundled, and the majority license is CC-BY-NC. |
| CSI / efficient-atom | A/B | Data-alignment and equal-budget method reference; no current experiment | TMLR paper and official scripts support CSI-based upstream selection and a 1/24-budget claim, but evaluated sources are the JMP atomic-property suite with no same-database PCQM Gap evidence. |
| GraphMVP | B | Geometry-teacher / 2D-student pretraining template; no current initialization | Contrastive 2D/3D positives plus variational representation reconstruction cleanly separate privileged geometry from downstream 2D inference, but there is no direct PCQM Gap result and the published geometry is not ETKDG. |
| GeoSSL-DDM | B | Pair-distance denoising objective reference; no current experiment | SE(3)-compatible multi-noise distance score matching has broad geometry-task evidence, but it is not a PCQM Gap result, requires all-pair distances, and assumes a geometry role that must be rebuilt for ETKDG. |
| UnifiedMolPretrain | B | Same-PCQM 2D/3D pretraining design reference; no current initialization | Public code and about 3.38M PCQM pretraining support symmetry-aware coordinate reconstruction and explicit 2D/3D task separation; no direct PCQM Gap score and no current ETKDG contract. |
| VideoMol | B | Multi-view geometry preprocessing reference only; no current experiment | About 2M PCQM molecules and a public 60-frame rotating-video pipeline make the artifact auditable, but rendering/storage cost, RDKit/MMFF/PyMOL dependence, and no direct PCQM Gap score block the graph route. |
| GraphGPT | B | No current experiment; completed artifact/sequence reference | Public source and four PCQM checkpoints plus README metrics; graph-to-sequence and 3D geometry contracts are not the current ETKDG path. |
| MetaGIN | B | Compact direct-PCQM comparator; no current experiment | The paper reports `0.0851` with `8.87M` parameters and 1/2/3-hop path-count ablations; the official repository links a Zenodo checkpoint. The static hop features are a 2D proxy, the public license surface was not closed, and the split/role is not the sealed MolGap comparison. |
| AUTAUT | B/C | Auxiliary-task selection algorithm reference only; no current experiment | NeurIPS 2025 paper uses LLM retrieval/selection and gradient-alignment adaptive weighting across nine MoleculeNet tasks, with PCQM4Mv2 only in a pretrain--fine-tune comparison; the paper-reported code URL returned 404 during audit and no direct PCQM Gap result is shown. |
| SubgDiff | B | Substructure-aware denoising reference; no current experiment | NeurIPS 2024 paper, MIT code, and Zenodo data expose subgraph prediction, expectation-state diffusion, and k-step same-mask corruption with PCQM4Mv2 pretraining; downstream evidence is MoleculeNet/MD17 and the geometry/runtime are not ETKDG-compatible. |
| UniGEM | B | Loss-scheduling and joint-generation reference only; no current experiment | ICLR 2025 paper and MIT code expose two-phase nucleation/growth diffusion and `homo/lumo/gap` switches, but property evidence is QM9, the PCQM link is a Frad comparison, and explicit 3D generation is outside the current route. |
| Hyformer | A/B | Alternating-mask joint-objective reference only; no current initialization | TMLR 2026 paper, BSD-3-Clause code, and public 8M/50M weights expose a shared Transformer for causal LM, bidirectional MLM, and property prediction, but the 19M external corpus, MoleculeNet targets, and no-ETKDG role manifest do not match PCQM4Mv2 Gap. |
| LAC | B | Same-database train-only relative-loss reference; no current experiment | ICLR 2025 ablations support hard-pair curriculum and relative prediction loss, but there is no direct PCQM4Mv2 result or verified implementation; pair mining requires a frozen train-only manifest and leakage audit. |
| ST-KD | B | Graph-to-SMILES compression reference only; no current initialization | The primary paper supplies structure-tokenized SMILES, virtual-token feature transfer, and attention-bias distillation with a historical PCQM4M-LSC validation ablation, but no matched PCQM4Mv2 role or independently retrievable official code/checkpoint. |
| LW-MPP | B | Paper-only compression/distillation reference; no current experiment | The Journal of Nanjing University paper combines attention/value relation transfer with Fisher structured pruning and reports three historical PCQM4M-LSC validation seeds, but no official artifact or closed PCQM4Mv2 mapping. |
| MolPeg | B | Same-database train-role data-selection reference; no current experiment | NeurIPS 2024 source-free pruning compares an online model with an EMA reference and uses PCQM4Mv2 for upstream GraphMAE/GraphCL pretraining, but it reports downstream HIV/PCBA/QM9 rather than PCQM Gap and has no official artifact. |
| KPGT | A/B | Public knowledge-node pretraining reference; no current initialization | Nature Communications paper plus Apache-2.0 code, data/splits, pretrained and downstream weights; external ChEMBL29, roughly 100M LiGhT model, no direct PCQM4Mv2 frontier result, and no ETKDG role. A same-database descriptor/fingerprint objective is the only portable hypothesis. |
| KGG | A/B | Electronic-proxy auxiliary-pretraining reference; no current initialization | JCIM paper, MIT code, and public ZINC15/ChEMBL29 model surface expose 11 hybridization/bond/adjacency/property pretexts and QM7/QM8/QM9 evidence. The “orbital” inputs are deterministic proxies, not quantum-orbital labels, and no direct PCQM Gap result is reported. |
| GraphFP | A/B | Fragment-level pretraining reference; no current experiment | NeurIPS paper and public code/data/weights expose mined fragments, contrastive fragment alignment, and predictive fragment tasks. Its 456K ChEMBL/MoleculeNet results have no direct PCQM Gap or ETKDG evidence; a future adaptation must mine fragments train-role-only. |
| MoleVers | A/B | Frontier-property source-task pretraining reference; no current initialization | Primary paper and official stage scripts expose masked-atom/denoising pretraining followed by HOMO/LUMO/dipole supervision. GDB17/Psi4/RDKit geometry, no direct PCQM Gap result, no verified released checkpoint/data package, and paper/script noise mismatch block direct use. |
| Supervised energy pretraining (Gao et al.) | B | Energy/geometry teacher reference only; no current experiment | PubChem PM6 energy pretraining and gradient regularization are mechanistically useful, but PM6 geometry/theory is not B3LYP/6-31G*/ETKDG, no direct frontier source task is shown, and no author code/checkpoint closes transfer. |
| PM6-ML | A/B | Completed Delta-learning/artifact reference; no current experiment | Peer-reviewed residual formula, MOPAC wrapper, model files, explicit SPICE/NCIAtlas pool, and 40-seed selection provide a complete energy Delta workflow. It has no scalar Gap result and all target/theory/conformer roles are external. |
| O_SMI-SSM-336M | A/B | Foundation-model/inference-efficiency reference; no current initialization | Nature paper, IBM code/notebooks, and Hugging Face weights expose a 91M-PubChem-SMILES Mamba foundation model and HOMO/LUMO speed study. External corpus, 336M scale, no matched PCQM Gap result, and SMILES-only input contract block direct use. |
| HieGT | A/B | Direct-PCQM motif-hierarchy reference; no current experiment | Open primary paper reports deterministic motif decomposition, AGA/MGA ablation, and `0.0769/0.0781` validation/test-dev values, but uses DFT/RDKit geometry and has no independently closed code/checkpoint. |
| MolCHG | B | Same-database fragment/source-task pretraining candidate after K3; no current initialization | Primary paper and public repository expose atom/bond/fragment/graph hierarchy, exact 0.2/0.4/0.4/0.4 loss balance, 250K-ZINC15 recipe, and ablations. No direct PCQM Gap result, external ZINC rows, missing bundled .pt files, and unverified license surface require a train-role-only rebuild and scratch/no-auxiliary/frozen-probe controls. |
| BiScale-GTR | A/B | Same-database Graph-BPE/masked-fragment candidate after K3; no current initialization | Primary paper and MIT repository expose validity-filtered WL fragments, OOV recursion, atom/fragment gated fusion, masked-fragment Transformer, 800-token recipe, and documented checkpoint paths. No PCQM Gap result, external ChEMBL roles, and un-hashed checkpoint paths require tokenizer/round-trip acceptance before compute. |
| FragNet | A/B | Fragment hierarchy and attribution reference only; no current experiment | Primary paper and public PNNL repository expose BRICS atom/bond/fragment/connection graphs, updated bond-to-atom edges, masked-fragment attribution, and runnable workflows. Uni-Mol-derived pretraining and an RDKit 3D path are outside the current ETKDG contract, with no direct PCQM Gap result. |
| MORE | A/B | Multi-view pretraining control reference; no current initialization | AAAI paper and MIT code expose masked atoms, MACCS subgraphs, RDKit graph descriptors, pair distances, explicit 4.5/5.0/1.0/0.04 weights, linear-probe/fine-tune controls, and MORE.pth. ZINC2M and random/MMFF conformers are external; only a separately contracted ETKDG same-database adaptation is admissible. |
| FragmentNet | B/C | Learned fragment-tokenizer inspiration only; no current experiment | Primary paper exposes adaptive pairwise merging, WL-aware hashed fragments, VQVAE codebooks, and masked-fragment sequence modeling with ten-seed ablations. No independently verified official code/checkpoint, no PCQM Gap result, and no ETKDG route block implementation. |
| PG-MLD | B/C | Dynamic 3D teacher and 3D-to-1D distillation method reference only; no current experiment | BioRxiv full text and public code expose equivariant frame encoding, Liquid Time-Constant modeling, perturbation/trajectory losses, frozen atom/molecule alignment, and atomic checkpoint logic. External OpenMM/ZINC/charge roles, missing bundled assets, no direct PCQM Gap result, and no ETKDG contract make it a low-priority rewrite rather than a drop-in teacher. |
| ConforFormer | A/B | Conformer-invariance and frozen-probe reference; possible post-selection ETKDG adaptation only | Published Digital Discovery paper plus MIT code expose two-view NT-Xent, exact `d=512/n=128/tau=0.07` settings, deterministic conformer sampling, data pipelines, results, and a named HF checkpoint. Uni-Mol/MMFF/OMol geometry, no direct PCQM Gap result, and unexplained distance-normalization constants require a fresh ETKDG pair audit; no external weights or conformers are imported. |
| CardinalGraphFormer | B/C | Sparse support-cardinality mechanism reference; no current initialization | Full arXiv paper gives a query-gated unnormalized sum over the same `K=3` support as normalized attention, matched no-CPA and explicit-size controls, five-seed OGB/ADMET results, and a 28M-corpus recipe. No direct PCQM Gap result, external ZINC20/ChEMBL35 corpus, 2D-only contract, and unavailable reproducibility package prevent direct use. |
| Contrastive KERMT | B/C | Probabilistic graph-to-SMILES pretraining reference; no current initialization | Full arXiv paper gives cMIM graph-to-SMILES reconstruction, in-batch mismatched negatives, combined KERMT local targets, cMIM-only ablation, frozen probes, and split/statistical details. No direct PCQM Gap result, external/transductive ADME corpora, anonymized code, and high compute require a separately authorized same-database rewrite. |
| GraphQPT | B/C | Electronic-teacher design and representation-analysis reference only | MIT code/data package separates atom-level QM, graph-level PCQM HLG, masking, and scratch controls; its external theory/geometry and ADMET/HLM downstream do not match current PCQM/ETKDG. |
| PCQM 3D-prior distillation | B/C | Negative smoke/control reference only | Explicit EGNN teacher, 2D student, KD/feature losses, ETKDGv3+UFF option, manifests, and paired bootstrap are useful; the checked-in synthetic smoke shows no positive distillation gain. |
| TGT | A/B | No current experiment; completed teacher/sanity reference | Public data, scripts, and weights with strong PCQM numbers, but learned/RDKit distance pipeline and 102M-scale stages are a different geometry/compute contract. |
| GPS++ PCQM | A/B | No current experiment; IPU systems reference | Public code/checkpoints and PCQM scaling evidence, but custom Graphcore ops and accelerator-specific packaging exceed the bounded screen. |
| Pre-training via Denoising (PVD) | A/B | No current experiment; future ETKDG protocol reference | Direct PCQM4Mv2 self-supervised upstream, MIT code/checkpoint, QM9 HOMO/LUMO/Gap transfer, and explicit random-init/Noisy-Nodes/pretraining controls; DFT coordinates and no same-contract PCQM Gap result block current initialization. |
| Fractional Denoising (Frad/FradNMI) | A/B | No current experiment; future chemical-aware ETKDG pretraining reference | Public ICML/NMI papers, MIT code/weights/data, RN/VRN chemical-aware noise, and QM9/force/robustness ablations; DFT/RDKit+MMFF geometry, QM9 downstream, legacy stack, and closed torsion route block current initialization. |
| SliDe | A/B | No current experiment; future force-consistent ETKDG pretraining reference | Public ICLR paper and MIT code/checkpoints; BAT bond/angle/torsion prior, random slicing, PCQM label-free pretraining, force audit, and QM9 frontier ablations are unusually complete. DFT coordinates, OpenFF/Sage prior, QM9 rather than PCQM Gap downstream, and estimator/model cost block direct initialization. |
| CCMD | B | No current experiment; future teacher-loss control template | Direct PCQM validation shows global molecular-token distillation can help, while naive local atom-token loss hurts and size-coordinated local loss repairs it. DFT teacher geometry, non-audited split wording, large model scale, and absent verified code/checkpoint require a separate ETKDG contract. |
| Denoise-and-Distill (D&D) | B | Future frozen-teacher control; no current experiment | Exact PCQM pretraining and graph/node distillation provide direct evidence for global versus local teacher losses, but the teacher uses DFT coordinates and no official code/checkpoint was verified. |
| DelFTa | A/B | Same-PCQM proxy-delta protocol reference; no current experiment | Public direct-vs-residual HOMO/LUMO/Gap controls and cost accounting are strong, but QMugs/GFN2-xTB/DFT labels and geometry differ; a CPU residual-variance gate is mandatory before any adaptation. |
| NVIDIA PCQM4Mv2 winner | A/B | No current experiment; delivery-time ensemble/OOF reference | Completed direct PCQM challenge/validation code, heavy-atom shift diagnostic, 2D/3D/image diversity, KPGT-style regularization, and Huber OOF stacking. Provided SDF/RDKit image views, train+valid fold construction, 39-checkpoint compute, and sealed-role mismatch block direct MolGap comparison. |
| 3D-PGT | B | Historical direct-PCQM reference; conditional future contract only | `0.0762` validation evidence and public MIT code, but 42.6M DFT-pretrained GPS does not fit the bounded ETKDG screen. |
| AniDS | B | Design reference only | Public PCQM label-free pretraining and anisotropic noise, but 129M force-field model, DFT geometry, and no direct PCQM Gap result. |
| 3D-EMGP | B | Design reference only | Public physical denoising objective and QM9 orbital fine-tuning, but GEOM-QM9/MD17 rather than PCQM. |
| Mol-MFFGE | B | Task-aware denoising reference only | Public paper/code and QM9 `homo/lumo/delta` path, but GEOM/SPICE/MD17 geometry and no direct PCQM Gap result. |
| OCNet | B | Yes, external teacher/OOD route after identity audit | Strong conjugated-domain paper/code/data/weight package, but theory, geometry, dimers, and labels differ from PCQM. |
| nablaColors-3D / UniProp | B | Geometry/artifact reference only | Peer-reviewed optical benchmark with public LMDB/splits/manifests/checkpoint hashes and explicit low-cost-to-DFT refinement; experimental targets, solvent/theory roles, and ETKDG2 geometry differ from MolGap. |
| Conjugated-polymer D-MPNN pretraining | B | External domain-matched pretraining reference only | Published three-way transfer comparison is relevant, but targets and geometries are external; cited artifact URL currently returns `404` in the audit. |
| DFT-feature-assisted optical-gap transfer | B | Delta/teacher protocol reference only | Public MIT code/data and full paper/SI are available, but it predicts experimental optical gaps from an oligomer DFT feature rather than a same-label residual. |
| Frontier-orbital Chemprop transfer | C | No current experiment | Published abstract/preview reports a promising GFN2-xTB-trimer transfer pattern, but the full reproducibility packet is not available in the audited sources. |
| OSCAgent graph--SMILES/LUMO pretraining | C/B | Objective reference only; no current experiment | Primary paper reports a symmetric graph--SMILES InfoNCE objective, auxiliary LUMO regression, and external PCE ablations, but no PCQM Gap result, closed theory/geometry contract, official code, or checkpoint. |
| QuADMET-Former | A/B | Target-bundle and charge-conservation teacher design only; no current experiment | Public MIT code, HF checkpoint surface, and exact QMugs config expose Löwdin charge, E3FP, dipole, Gap, and total-energy pretraining; QMugs geometry/theory and ADMET downstream do not match PCQM/ETKDG. |
| BOA | A/B | Electron-density teacher architecture/reference only; no current experiment | ICLR 2026 basis-overlap model has public code, checkpoint surface, and explicit QM9 VASP/PySCF density results plus QMugs size transfer, but density targets and external theory/geometry do not match current B3LYP/6-31G*/ETKDG HOMO/LUMO/Gap. |
| C-FREE | A/B | Non-contrastive ego-net/predictor objective reference only; no current experiment | ICML 2026 paper, MIT code, and public checkpoints support an EMA-target local-to-global pretraining design; GEOM/RDKit conformers and QM9 grouped frontier evidence do not establish current PCQM/ETKDG benefit. |
| OneQMC/Orbformer | A/B | Public wavefunction/density teacher feasibility reference only; no current experiment | MIT OneQMC code, LAC data, `lac.chkpt`, and NERD extraction support an ab initio electronic teacher concept, but the JAX/A100 cost, LAC/NIST geometry, light-element scope, zero-shot warning, and absent scalar PCQM Gap result block current use. |
| OrbitAll | B | Orbital-feature delta-learning method reference only; no current experiment | QM9star and OMol25 evidence supports a three-way direct/scalar-residual/orbital-residual comparison, but no verified code/checkpoint is released and the low-level QM/theory/geometry contract is external. |
| NN-xTB | A/B | Hamiltonian-level low-fidelity teacher and residual-cost reference only; no current experiment | Code Ocean-backed source/model/script provenance and strong energy/force/frequency evidence support a self-consistent teacher design, but no current frontier benchmark or ETKDG/PCQM contract is established. |
| Message-Passing Delta-ML | A/B | Organic-electronics excited-state Delta protocol reference only; no current experiment | Public ZINDO-to-TDDFT S1 dataset/models/tutorials and electronic descriptors support residual and core-aware split design, but S1/oscillator targets and geometry/theory roles are external to PCQM Gap. |
| OPoly26 | B | External polymer database/packaging audit only | Public data/code and paper are available, but the theory/field metadata discrepancy and non-ETKDG polymer geometry block current teacher use. |
| PubChemQC-100K -> CO-610 transfer | B | Same-source transfer protocol reference only | Published direct-vs-transfer comparison and public ESI are useful, but external target/geometry, absent code/checkpoint, and source overlap block current initialization. |
| GFN2-xTB/COCONUT proxy workflow | A/B | CPU proxy and delta-audit reference only | Complete MIT GitHub/Zenodo workflow and data are available, but xTB labels, RDKit/xTB geometry, and COCONUT roles do not match PCQM/B3LYP/ETKDG. |
| QMCVNet / PubChemQC PM6-to-B3LYP voxel control | B | Historical low-fidelity geometry/rotation reference only | Same-lineage paper and SI expose PM6/MMFF controls, but the coordinate mix, preprint status, no code/checkpoint, and non-residual target block current use. |
| QUED electronic descriptor | A/B | Teacher-feature and ablation reference only | Strong paper/code/model/data packet, but DFTB3+MBD, QM7-X/ADMET roles, and CREST/MMFF geometry do not match current B3LYP/6-31G*/ETKDG targets. |
| POS-EGNN/OMol25 | B | Multi-task physical-consistency and 3D-foundation reference only | Strong paper/code/weight surfaces, but the paper's frontier model is external OMol25/MD/ωB97M-V and the public weights are MPtrj energy/force/stress weights. |
| AEGCNN-MTL | B | Task-grouping and negative-transfer reference only | Peer-reviewed QM9 evidence is detailed, but code and BDG data are not public and no current PCQM/ETKDG result is shown. |
| LUMIA | A/B | Organic-domain pretraining and explanation engineering reference only | Formal JCTC paper plus MIT/Zenodo assets, but no matched PCQM Gap result; the released data, RGCN checkpoint, and knowledge masks are external roles. |
| OSCs_RGGN | C | No | Repository claim lacks a reproducible data/target/license/split packet. |
| GLACIER | B | Teacher engineering reference only | Public multi-teacher distillation and checkpoint; no PCQM Gap result, external Enamine/TDC/MoleculeNet role. |
| ChemBERTa-3 | B | Engineering reference only | Public foundation-model pipeline and artifacts; no matched PCQM Gap evidence and external corpus roles. |
| ChemFM | B/C | No current candidate | 3B-scale UniChem SMILES foundation model; no ETKDG or direct PCQM Gap contract. |
| GPSE | B | Conditional frozen structural teacher after provenance audit | Public PSE encoder and PCQM-named checkpoint, but no direct Gap evidence and checkpoint lineage requires verification. |
| CondPSE | B | No current candidate | Synthetic expressivity improves, but molecular-property gains are not consistently better than GPSE and no code was verified. |
| GCPE | C | No | Publisher-level 2026 PCQM claim lacks exact metric, split, code, and checkpoint in the audited surface. |
| Chemprop benchmark v2 | B | Engineering/sanity reference only | Completed MIT benchmark package with PCQM/QM9 Gap scripts and explicit split artifacts, not a new MolGap method. |
| ECMMR | C | No | Abstract-level PCQM pretraining claim without an auditable metric/code/checkpoint packet. |
| QM9 embedding KD scalability study | B | Teacher-loss/control reference only | Open-access paper and code include QM9 HOMO/LUMO/Gap teacher training, smaller students, and cross-domain transfer; no PCQM/ETKDG result. |
| EDG electron-density teacher | B | Teacher protocol only, after identity/theory/geometry audit | IJCAI paper, MIT code, public ImageED/ED-teacher artifacts, and QM9 HOMO/LUMO/Gap ablations; PCQM-derived 2M source and non-ETKDG/non-matched basis require a new contract. |
| D&D 3D teacher to 2D student | B | Yes, future teacher template | Exact PCQM pretraining and clear role separation; DFT coordinates and no official code. |
| DelFTa-style same-PCQM proxy delta | B | Conditional only | Strong open-source delta evidence, but target/geometry/data differ; CPU residual gate is mandatory. |
| MolSpectra electronic-spectrum teacher | B | Conditional, post-selection only | Strong QM9 electronic ablation, but QM9S theory/geometry/license differ from PCQM and no direct PCQM Gap result is exposed. |
| Charge/bond/electron teacher | B | Conditional, source by source | Useful electronic signal, but external-label overlap and theory level must be audited. |
| DeMol | C | No | Exact-PCQM claim but no usable code/weights, oversized model, and target labels in pretraining. |
| DGT transplant | B | No as a queue item | Public paper, MIT code, Zenodo archive, and Figshare data establish a complete pretraining/dual-graph reference, but no direct PCQM Gap leaderboard score or visible PCQM checkpoint/config is exposed; QM9/DFT/MMFF/UFF geometry and EdgeState overlap block direct use. |
| MolCL-SP | B | Multimodal substructure-pretraining reference; no current initialization | Same-PCQM paired pretraining and non-overlapping perturbations are useful controls, but the reported frontier results are QM9, the 3D lineage is not ETKDG, and the public weight/data surface is not versioned. |
| SpaceFormer | B | Sparse 3D occupancy-token reference; no current experiment | Occupied/empty space modeling and coordinate reconstruction are distinctive, but the 19M external corpus, Hartree/GDB-17 task, and 67.8M/8-A100 contract are external. |
| MFGP-GEM | A | Multi-fidelity/delta protocol reference; no current experiment | Direct, delta, and autoregressive controls are unusually complete on a small benzoquinone benchmark, but theory, geometry, and domain differ from PCQM/ETKDG. |
| Multi-fidelity GNN transfer | A | Low-to-high strategy matrix reference; no current experiment | Predictions, embeddings, pretrain/fine-tune, readout-only, and transductive/inductive controls are valuable, but QMugs GFN2-xTB/DFT roles and conformers are external and may leak identities. |
| GAPE | A/B | Future train-role-only PE audit; no current experiment | Topology-matched graph-alignment PE has strong small-subset evidence, but the edge-feature-free contract and transductive pretraining question require frozen/joint/no-PE controls. |
| GRPE | A/B | Compact relative-value/context reference; no current experiment | Direct PCQM4Mv2 evidence and official code are strong, but the public models are large and the old stack needs local acceptance. |
| TetraGT | A/B | Higher-order geometry upper bound; no current experiment | Direct PCQM evidence supports explicit angle/torsion states, but MMFF-style coordinates, large models, and unavailable runnable code block the current route. |
| GEM-2 | A/C | Factorized many-body reference; no current experiment | Direct `0.0793/0.0806` PCQM evidence is strong, but MMFF94 geometry and all-pair higher-order tensors conflict with ETKDG and budget. |
| TokenGT | A/B | Large 2D token/PE reference; no current experiment | Direct test-dev evidence and public code isolate graph identifiers, but the 48.5M full Transformer is not a bounded local adaptation. |
| GPTrans | A/B | Node--edge propagation reference; no current experiment | Public PCQM configs/checkpoints and explicit propagation ablations are strong, but the large best model overlaps EdgeState and requires a new bounded hypothesis. |
| Edge Transformer | B/C | Higher-order theoretical reference only | The compact `(2,1)` idea is informative, but unrestricted pair states are cubic and the PCQM comparison is single-seed validation-only. |
| (2,1)-GT / WL Transformers | A/B | Compact higher-order 2D reference; no current experiment | O(n+m) tuple tokens avoid full pair-state cost, but the PCQM number is single-seed validation and target-supervised rather than label-free pretraining. |
| GDT | A/B | PE-control and standard-attention reference; no current experiment | NeurIPS 2025 evaluates NoPE/LPE/SPE/RWSE/RRWP on full PCQM with three seeds and explicit scale controls, but no independently closed code/checkpoint or test-dev result was found and dense attention bias is costly. |
| QuantumCanvas | B | Electronic-teacher and two-body artifact reference only; no current experiment | Public paper/code/Zenodo bundle, composition-held-out benchmark, and QM9 transfer are useful, but SCC-DFTB/PTBP diatomics, explicit coordinates, schema discrepancy, and no PCQM result block current use. |
| TMC-Delta-ML | A/B | Delta-learning protocol and cost-control reference only; no current experiment | Peer-reviewed paper, MIT code, Zenodo low-fidelity graphs, fixed CSD-ID splits, and explicit HOMO--LUMO residual modes are unusually complete, but tmQMg transition-metal chemistry, u-NatQG/GFN2-xTB geometry, and non-PCQM theory block current use. |
| SelectedML | A/B | Chemical-family stratification and direct-versus-Delta control reference; no current experiment | Peer-reviewed QM7/QM9 evidence plus public CM/BoB/SLATM/KRR scripts support a family-conditioned learning-curve audit, but the source theory, representation, and coordinate roles do not match PCQM/ETKDG. |
| HLP-Stack | B/C | Negative descriptor/feature-provenance audit only; no current experiment | Public paper/repo and raw/model artifacts are useful, but the near-perfect QM9 results rely on DFT-derived 3D descriptors at B3LYP/6-31G(2df,p), which are target-adjacent and not the current SMILES/ETKDG input contract. |
| PET-MAD-DOS | A/B | Completed DOS/electronic-teacher packaging reference only; no current experiment | Public paper, Materials Cloud data/scripts, UPET implementation, and Hugging Face model are unusually complete, but PBEsol/MAD/periodic roles and DOS-derived gap do not match B3LYP/6-31G* PCQM HOMO/LUMO/Gap. |
| FieldMACE | A/B | Long-range multipole-message and foundation-transfer reference only; no current experiment | Public paper/code/checkpoints/Figshare data provide reproducible QM/MM energy/force evidence, but explicit MM point charges, solvated geometries, and no frontier target violate the current ETKDG/PCQM contract. |
| MACE-POLAR-1 | A/B | Charge/spin-constrained foundation-teacher design only; no current experiment | Public ASL checkpoints and paper-level OMol25 training are complete, but the model has no PCQM frontier head/result and its hybrid-DFT 3D contract is external. |
| MACE-H | A/B | Operator-level electronic/high-body-order reference only; no current experiment | Public MIT code and Zenodo configs reproduce periodic KS Hamiltonian prediction, but OpenMX/FHI-aims materials, basis/SOC, and reciprocal-space roles are not molecular PCQM/ETKDG Gap evidence. |
| CheMeleon | A/B | Descriptor-pretraining and checkpoint-transfer reference only; no current experiment | Public paper/code/data/weights establish a complete D-MPNN route, but PubChem/Mordred/SMILES pretraining, no matched PCQM Gap result, no ETKDG path, and high foundation cost block direct use. |
| Zatom-1 | A/B | 3D foundation/teacher and frozen-trunk/LoRA control reference only; no current experiment | Public code/checkpoints and QM9 `homo`/`lumo`/`gap` route are complete, but QM9/Matbench/OMol25/MPtrj, explicit 3D coordinates, no PCQM matched result, and foundation-scale cost block direct use. |
| OrbNet-Equi | A/B | AO-electronic Delta-learning and direct-versus-residual control reference only; no current experiment | Public PNAS/Zenodo source surface and frontier-property crossover evidence are strong, but GFN-xTB operator features, QM9/SDC21 theory/geometry, no matched PCQM result, and external archive block direct use. |
| Image-super-resolution density | A/B | Real-space density-teacher and post-diagonalization frontier-property reference only; no current experiment | Public paper/model/data package reports QM9 one-step HOMO/LUMO/Gap and geometry-transfer controls, but PBE/GTH grids, density labels, one-step diagonalization, and no PCQM/ETKDG direct head block current use. |
| 3DGrid-VQGAN | A/B | Density-grid foundation and frozen-teacher reference only; no current experiment | Public IBM/Hugging Face artifact and QM9 frontier readouts are useful, but RHF/STO-3G/MINDO3 grids, QM9 theory, unit-ambiguous table values, external PubChem rows, and multi-terabyte storage block current use. |
| Graph2Mat | A/B | Sparse operator-teacher and self-consistency/UQ reference only; no current experiment | Public code/data and SCF diagnostics are strong, but SIESTA basis/pseudopotentials, external coordinates, matrix labels, and no direct frontier result block current use. |
| ACE density matrix | A/B | Grassmann/projector-constrained operator/UQ reference only; no current experiment | Public Julia code and DaRUS assets provide reproducible operator controls, but solvent/QM-MM/basis-dependent 3D density frames and no PCQM/ETKDG frontier result block current use. |
| SMILESDFT-CLIP/SigLIP | A/B | Multimodal pretraining and rotation/invariance-control reference only; no current experiment | Public IBM code/checkpoints are useful, but PubChem density pairs, external grids, no matched PCQM frontier result, and external weights block current initialization. |
| QMLearn | A/B | Gamma/δ 1-RDM electronic-teacher and operator-readout reference only; no current experiment | Public paper/source/Zenodo package exposes matrix residuals and post-Fock frontier readouts, but molecule-specific GTO/PySCF/Eckart roles and no PCQM/ETKDG result block direct use. |
| QMLearn-SCF | A/B | SCF-threshold, residual, force and geometry-coverage teacher reference only; no current experiment | Peer-reviewed follow-up and Zenodo archive provide explicit KRR/linear-δ, force-correction and AIMD controls, but external GTO/B3LYP data, post-diagonalization Gap and 1.3-TB archive block current use. |
| STRUCTURES25 | A/B | Variational density/energy auxiliary-teacher reference only; no current experiment | Public equivariant orbital-free DFT code/models and perturbed-potential data route are strong, but PBE/QMugs density coefficients, explicit 3D basis contract, no direct frontier head and external license block current use. |
| KineticNet | A/B | Derivative-aware orbital-free electronic-teacher design only; no current experiment | The paper provides a complete E(3)-equivariant grid architecture, off-ground-state perturbation strategy, and held-out metrics, but BLYP/cc-pVDZ, real-space grids, author-request data, and no frontier head block current use. |
| M-OFDFT | A/B | Residual KEDF, projected-gradient teacher, and pretrain/fine-tune protocol reference only; no current experiment | Nature/Zenodo/Figshare evidence is unusually complete, but PBE/6-31G(2df,p), QM9/QMugs/MD17/chignolin density coefficients, explicit 3D geometry, and no direct PCQM frontier head block current use. |
| QH9 / ∇²DFT / nablaDFT | A/B | Public operator-data and acceptance reference only; no current experiment | QH9/∇²DFT provide Hamiltonian/density schemas, loaders, and post-diagonalization metrics, but B3LYP/def2-SVP or ωB97X-D/def2-SVP, explicit 3D, and external identities block current import. |
| Self-Consistency Training / DEQHNet / NeuralSCF | A/B | Fixed-point and SCF-residual method reference only; no current experiment | Differentiable eigensolver/density residuals and SCF trajectory training are strong evidence, but matrix/density labels and basis/geometry contracts are absent from current PCQM. |
| HamEvo | A/B | Highest-priority operator-Delta method reference; no current experiment | Public `Delta H`, fixed-point, Anderson/Broyden, and staged pretraining evidence is unusually complete, but it is not a scalar `Delta Gap` result and uses external B3LYP/def2-SVP/3D roles. |
| QHFlow2 | A/B | Frontier-aware public Hamiltonian teacher/probe; no current initialization | QH9/rMD17 checkpoints expose HOMO/LUMO/Gap after H diagonalization, but explicit 3D, B3LYP/def2-SVP, checkpoint licensing, and no matched PCQM/ETKDG route block transfer. |
| HELM | B | Hamiltonian-pretraining protocol reference only; no current artifact | Frozen versus fine-tuned energy heads improve low-data energy prediction in the paper, but code/data are TBA and no direct frontier/PCQM result is available. |
| GraphQPT atom-level quantum pretraining | B/C | Local-electronic teacher and pretraining attribution reference only; no current experiment | Peer-reviewed Graphormer study and public code compare atom-level QM, PCQM HLG, masking, and scratch, but downstream tasks are ADMET/HLM and the atom-level geometry/theory is external. |
| Tetrahedral Molecular Pretraining | A/B | Same-database 3D pretraining/teacher reference only; no current initialization | Public Pattern Recognition paper, MIT code/checkpoint, PCQM downstream comparison, and label-free tetrahedral objectives are strong, but DFT coordinates, large scale, legacy stack, unresolved validation lineage, and non-ETKDG inference block direct use. |
| Self-Conditioned Denoising | B | Evidence-only implementation/checkpoint audit and later ETKDG objective adaptation; no current initialization | Primary 2026 paper, official code, and PCQ checkpoint establish a clean-embedding-conditioned denoising route, but the published downstream evidence is QM9 after equilibrium-coordinate PCQ pretraining rather than direct PCQM Gap. |
| DenoiseVAE | B paper / C code | Paper/method and code-negative-control audit only; no current initialization | Direct PCQM validation number is useful, but the public builder/configuration violates ETKDG/GEOM boundaries and the code lacks a usable checkpoint manifest and contains an undefined return variable. |
| PySCFAD / differentiable ML-QM | A/B | Differentiable Hamiltonian-to-observable implementation reference only; no current experiment | Public package/tutorial and peer-reviewed paper establish direct, indirect, multi-target, and reduced-basis controls, but QM7/QM9, explicit basis/integrals, and no PCQM/ETKDG frontier route block current use. |
| SAP Hamiltonian residual | B | Operator-Delta design reference only; no current experiment | The paper reports `Delta F = F_PBE0 - F_SAP` and auxiliary orbital/density/dipole losses with strong QM9 numbers, but it has no verified public code/checkpoint and its SAP/SAIAO/PBE0/cc-pVDZ contract is external. |
| One-particle density-matrix teacher | A/B | Direct density-teacher concept only; no current experiment | The paper provides a small-system BLYP/cc-pVDZ DM teacher and SCF-initialization test, but no public production checkpoint/data and no PCQM/ETKDG scalar frontier route are available. |
| HamGNN | A/B | Completed equivariant Hamiltonian artifact reference only; no current experiment | Official code plus Zenodo models/data support reconstruction and spectrum checks, but tight-binding/materials/basis/explicit-3D roles do not match molecular PCQM Gap. |
| HAMSTER | A/B | Physics-informed Hamiltonian Delta packaging reference only; no current experiment | Public MIT Julia code/data and the paper support TB-to-DFT residual learning with band-gap acceptance, but periodic PBE/VASP/SOC materials are outside the current organic PCQM/ETKDG contract. |
| QHNetV2 / SPHNet / WANet | A/B | SO(2), sparse-gate, and overlap-aware loss efficiency reference only; no current experiment | Public operator results do not establish a same-contract scalar Gap gain; basis-dependent matrices and explicit 3D remain external. |
| Meyer derivative training | C | Negative control for derivative supervision only; no current experiment | The open 1D study demonstrates both derivative-loss gains and noisy-unconstrained-optimization failure; it has no molecular, frontier, PCQM, or ETKDG evidence. |
| Weak README-only PCQM projects | C | No | No independent provenance or comparable evidence. |

No remote job, external dataset merge, pretrained initialization, or production
change was made while writing this audit.

## Primary sources

- [OGB PCQM4Mv2 specification](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
- [Self-Conditioned Denoising paper](https://arxiv.org/html/2603.17196v1), [official code](https://github.com/TyJPerez/SelfConditionedDenoisingAtoms), [PCQ checkpoint](https://huggingface.co/Ty-Perez/ct-scd-pcq)
- [DenoiseVAE paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html), [paper PDF](https://openreview.net/attachment?id=ym7pr83XQr&name=pdf), and [code](https://github.com/liuyurou1/DenoiseVAE)
- [IEM IJCAI paper](https://www.ijcai.org/proceedings/2024/675), [PDF](https://www.ijcai.org/proceedings/2024/0675.pdf), and [official code](https://github.com/HongxinXiang/IEM)
- [MolInteract PAKDD paper PDF](https://kdd2025.kdd.org/wp-content/uploads/2025/07/paper_22.pdf) and [NeurIPS workshop record](https://neurips.cc/virtual/2024/102819)
- [LeJEPA arXiv record](https://arxiv.org/abs/2609.04261)
- [Does GNN Pretraining Help Molecular Representation?](https://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html), [paper PDF](https://papers.nips.cc/paper_files/paper/2022/file/4ec360efb3f52643ac43fda570ec0118-Paper-Conference.pdf), and [supplement](https://papers.nips.cc/paper_files/paper/2022/file/4ec360efb3f52643ac43fda570ec0118-Supplemental-Conference.pdf)
- [ET-OREO / May the Force be with You](https://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html), [paper PDF](https://papers.nips.cc/paper_files/paper/2023/file/e637029c42aa593850eeebf46616444d-Paper-Conference.pdf), and [OpenReview PDF](https://openreview.net/pdf?id=Ge8Mhggq0z)
- [JMP ICLR record](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a896a8bb065774169d9ad65f19208b7-Abstract-Conference.html), [paper PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/4a896a8bb065774169d9ad65f19208b7-Paper-Conference.pdf), and [official code](https://github.com/facebookresearch/JMP)
- [CSI paper](https://arxiv.org/abs/2502.11085), [TMLR/OpenReview record](https://openreview.net/forum?id=jfD9BsrDTb), and [official code](https://github.com/Yasir-Ghunaim/efficient-atom)
- [GraphMVP paper](https://arxiv.org/html/2110.07728) and [official code](https://github.com/chao1224/graphmvp)
- [GeoSSL-DDM paper](https://arxiv.org/html/2206.13602)
- [Unified 2D and 3D Pre-Training paper](https://arxiv.org/html/2207.08806) and [official repository](https://github.com/teslacool/UnifiedMolPretrain)
- [VideoMol paper](https://www.nature.com/articles/s41467-024-53742-z), [official code](https://github.com/HongxinXiang/VideoMol), and [processed-data mirror](https://github.com/ChengF-Lab/VideoMol)
- [3D-GSRD paper](https://arxiv.org/abs/2510.16780) and [official code](https://github.com/WuChang0124/3D-GSRD)
- [3D-MolT5 paper](https://arxiv.org/html/2406.05797) and [official code](https://github.com/QizhiPei/3D-MolT5)
- [MolSpectra paper](https://arxiv.org/html/2502.16284) and [official code](https://github.com/AzureLeon1/MolSpectra)
- [3D-PGT paper](https://arxiv.org/html/2306.07812) and [official code](https://github.com/LARS-research/3D-PGT)
- [AniDS paper](https://arxiv.org/html/2510.22123) and [official code](https://github.com/ZeroKnighting/AniDS)
- [3D-EMGP paper](https://ojs.aaai.org/index.php/AAAI/article/view/25978) and [official code](https://github.com/jiaor17/3D-EMGP)
- [Mol-MFFGE paper](https://www.sciencedirect.com/science/article/pii/S0031320325001918) and [official code](https://github.com/Yufei-Luo/Mol-MFFGE)
- [Pre-training via Denoising paper](https://arxiv.org/html/2206.00133), [official code/checkpoint](https://github.com/shehzaidi/pre-training-via-denoising), and [PCQM configuration](https://github.com/shehzaidi/pre-training-via-denoising/blob/main/examples/ET-PCQM4MV2.yaml)
- [Frad ICML paper](https://arxiv.org/html/2307.10683), [Frad code](https://github.com/fengshikun/Frad), [Frad NMI paper](https://arxiv.org/html/2407.11086), [FradNMI code](https://github.com/fengshikun/FradNMI), [released models](https://zenodo.org/records/12697467), and [source data](https://doi.org/10.6084/m9.figshare.25902679.v1)
- [SliDe paper](https://arxiv.org/html/2311.02124) and [MIT code/checkpoints](https://github.com/fengshikun/SliDe)
- [D&D paper page](https://ojs.aaai.org/index.php/AAAI/article/view/31986), [D&D PDF](https://ojs.aaai.org/index.php/AAAI/article/download/31986/34141)
- [Coordinating Cross-modal Distillation](https://arxiv.org/abs/2211.16712)
- [Denoise-and-Distill paper page](https://ojs.aaai.org/index.php/AAAI/article/view/31986) and [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/31986/34141)
- [NVIDIA Heterogenous Ensemble paper](https://arxiv.org/abs/2211.11035) and [MIT implementation](https://github.com/jfpuget/NVIDIA-PCQM4Mv2)
- [ViSNet PCQM technical report](https://ogb.stanford.edu/paper/neurips2022/pcqm4mv2_ViSNet.pdf)
- [DelFTa code](https://github.com/josejimenezluna/delfta), [DelFTa documentation](https://delfta.readthedocs.io/en/latest/), [delta-QML paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9093086/)
- [DelFTa paper](https://pubs.rsc.org/en/content/articlelanding/2022/cp/d2cp00834c)
- [GW/DFT delta-learning paper](https://doi.org/10.1088/2632-2153/acf545)
- [DeMol paper](https://arxiv.org/html/2603.00568), [author-linked repository](https://github.com/LiuYunqing/DeMol)
- [DGT code](https://github.com/zhangsy-ryan/DGT), [DGT paper](https://www.nature.com/articles/s41467-026-75005-9), [Zenodo code archive](https://doi.org/10.5281/zenodo.20009509), and [Figshare source data](https://doi.org/10.6084/m9.figshare.30665129)
- [QuantumCanvas paper](https://arxiv.org/abs/2512.01519), [MIT code](https://github.com/KurbanIntelligenceLab/QuantumCanvas), and [Zenodo dataset](https://doi.org/10.5281/zenodo.20631934)
- [TMC-Delta-ML paper](https://doi.org/10.1002/chem.71487), [official code](https://github.com/uiocompcat/TMC-Delta-ML), [tmQMg](https://github.com/uiocompcat/tmQMg), and [Zenodo low-fidelity graphs](https://doi.org/10.5281/zenodo.18348669)
- [Selected Machine Learning paper](https://doi.org/10.1039/D2MA00742H), [arXiv record](https://arxiv.org/abs/2110.02596), and [public code](https://github.com/b3rn4rdm/SelectedML)
- [HLP-Stack paper](https://doi.org/10.1039/D5RA08007J), [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959570/), and [public repository](https://github.com/college-of-pharmacy-gachon-university/HLP_STACK)
- [PET-MAD-DOS paper](https://doi.org/10.1039/D5DD00557D), [UPET code](https://github.com/lab-cosmo/upet), [Materials Cloud record](https://doi.org/10.24435/materialscloud:gs-z7), and [Hugging Face model](https://huggingface.co/lab-cosmo/pet-mad-dos)
- [FieldMACE paper](https://doi.org/10.1038/s41524-026-02048-3), [official code](https://github.com/rhyan10/FieldMACE), and [Figshare archive](https://figshare.com/articles/dataset/Models_data_and_code_for_publication_Incorporating_Long-Range_Interactions_via_the_Multipole_Expansion_into_Ground_and_Excited-State_Molecular_Simulations_/28497857)
- [MACE-POLAR-1 paper](https://arxiv.org/html/2602.19411) and [official foundation release](https://github.com/ACEsuit/mace-foundations/releases)
- [MACE-H paper](https://arxiv.org/html/2508.15108), [MIT code](https://github.com/maurergroup/MACE-H), and [Zenodo package](https://doi.org/10.5281/zenodo.15223696)
- [CheMeleon paper](https://arxiv.org/html/2506.15792), [official code](https://github.com/JacksonBurns/chemeleon), [training data](https://doi.org/10.5281/zenodo.15733574), [model weights](https://doi.org/10.5281/zenodo.15426600), and [Chemprop fine-tuning documentation](https://chemprop.readthedocs.io/en/main/chemeleon_foundation_finetuning.html)
- [Zatom-1 paper](https://arxiv.org/html/2602.22251), [official code](https://github.com/Zatom-AI/zatom), and [Zenodo checkpoints](https://zenodo.org/records/19766997)
- [OrbNet-Equi paper](https://doi.org/10.1073/pnas.2205221119), [arXiv record](https://arxiv.org/abs/2105.14655), and [Zenodo source/data/code](https://zenodo.org/records/6568437)
- [Image-super-resolution density paper](https://doi.org/10.1038/s41467-025-60095-8), [Zenodo model/code record](https://doi.org/10.5281/zenodo.15226766), and [Figshare data](https://figshare.com/articles/dataset/Image_Super-resolution_Inspired_Electron_Density_Prediction/25365508)
- [3DGrid-VQGAN paper](https://openreview.net/pdf/51c97777a512d94b366ffd7ba0d8979eba14e3c8.pdf), [IBM code](https://github.com/IBM/materials/tree/main/models/3dgrid_vqgan), and [Hugging Face checkpoint](https://huggingface.co/ibm-research/materials.3dgrid_vqgan)
- [Graph2Mat paper](https://doi.org/10.1088/2632-2153/adc871), [official code](https://github.com/BIG-MAP/graph2mat), and [DTU data](https://data.dtu.dk/articles/dataset/MD17_data_for_graph2mat/26195285)
- [ACE density-matrix paper](https://doi.org/10.1039/D5DD00230C), [official code](https://github.com/ACEsuit/ACEdensitymatrix), and [DaRUS data](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419/DARUS-4902)
- [SMILESDFT-CLIP/SigLIP workshop record](https://neurips.cc/virtual/2025/126037), [IBM code](https://github.com/IBM/materials/tree/main/models/smilesdft_clip), and [Hugging Face model card](https://huggingface.co/ibm-research/materials.smilesdft-clip)
- [QMLearn paper](https://doi.org/10.1038/s41467-023-41953-9), [source](https://gitlab.com/pavanello-research-group/qmlearn), [training data](https://doi.org/10.5281/zenodo.7946420), and [code snapshot](https://doi.org/10.5281/zenodo.8269767)
- [QMLearn-SCF paper](https://doi.org/10.1021/acs.jctc.5c01564), [open preprint](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/68c9a9763e708a76498380b5/original/main.pdf), and [Zenodo archive](https://zenodo.org/records/17103131)
- [STRUCTURES25 paper](https://arxiv.org/html/2503.00443v2), [official code](https://github.com/sciai-lab/structures25), and [documentation](https://sciai-lab.github.io/structures25/)
- [KineticNet paper](https://doi.org/10.1063/5.0158275) and [arXiv HTML](https://arxiv.org/html/2305.13316)
- [M-OFDFT paper](https://doi.org/10.1038/s43588-024-00605-8), [arXiv PDF](https://arxiv.org/pdf/2309.16578), [Zenodo implementation](https://doi.org/10.5281/zenodo.10616893), and [Figshare collection](https://doi.org/10.6084/m9.figshare.c.6877432)
- [Derivative-aware orbital-free DFT paper](https://doi.org/10.1021/acs.jctc.0c00580) and [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7482319/)
- [QH9 paper](https://arxiv.org/html/2306.09549), [AIRS/QHBench](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9), and [Zenodo](https://zenodo.org/records/8274793)
- [nablaDFT](https://doi.org/10.1039/D2CP03966D), [∇²DFT](https://arxiv.org/html/2406.14347), and [AIRS](https://github.com/divelab/AIRS)
- [Self-Consistency Training](https://proceedings.mlr.press/v235/zhang24ak.html), [DEQHNet](https://github.com/Zun-Wang/DEQHNet), and [NeuralSCF](https://github.com/songfeitong/neuralscf)
- [HamEvo](https://arxiv.org/html/2606.14498), [official code](https://github.com/axdfhj/HamEvo_official), and [data](https://huggingface.co/datasets/ZJUSCL/hamevo-data)
- [QHFlow2](https://arxiv.org/html/2602.16897), [official code](https://github.com/seongsukim-ml/QHFlow2), and [QH9 checkpoint](https://huggingface.co/ksusu/QHFlow2-QH9)
- [HELM](https://arxiv.org/html/2510.00224), [QHNetV2](https://arxiv.org/html/2506.09398), [SPHNet](https://proceedings.mlr.press/v267/luo25l.html), and [WANet/WALoss](https://arxiv.org/pdf/2502.19227)
- [Graphormer atom-in-a-molecule quantum pretraining paper](https://doi.org/10.1186/s13321-025-00970-0) and [GraphQPT code](https://github.com/aidd-msca/GraphQPT)
- [Tetrahedral Molecular Pretraining paper](https://doi.org/10.1016/j.patcog.2025.112638) and [MIT code/checkpoint repository](https://github.com/sunyuancheng/Tetrahedral-Molecular-Pretraining)
- [Self-Conditioned Denoising paper](https://arxiv.org/html/2603.17196v1), [official code](https://github.com/TyJPerez/SelfConditionedDenoisingAtoms), and [PCQ checkpoint](https://huggingface.co/Ty-Perez/ct-scd-pcq)
- [DenoiseVAE paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html), [paper PDF](https://openreview.net/attachment?id=ym7pr83XQr&name=pdf), and [code](https://github.com/liuyurou1/DenoiseVAE)
- [Fully differentiable ML/QM Hamiltonian learning](https://doi.org/10.1021/acs.jctc.5c00522), [Atomistic Cookbook](https://atomistic-cookbook.org/examples/hamiltonian-qm7/hamiltonian-qm7.html), and [PySCFAD](https://github.com/fishjojo/pyscfad)
- [SAP Hamiltonian residual](https://arxiv.org/html/2606.12326)
- [One-particle density matrix prediction](https://doi.org/10.1021/acs.jctc.4c00042), [arXiv](https://arxiv.org/abs/2401.06533), and [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11171273/)
- [HamGNN](https://doi.org/10.1038/s41524-023-01130-4), [official code](https://github.com/QuantumLab-ZY/HamGNN), [models](https://doi.org/10.5281/zenodo.8147631), and [training data](https://doi.org/10.5281/zenodo.8157128)
- [HAMSTER](https://doi.org/10.1038/s41467-026-70865-7), [MIT Hamster.jl code](https://github.com/TheoFEM-TUM/Hamster.jl), and [Zenodo data](https://doi.org/10.5281/zenodo.18485403)
- [Generic molecular knowledge-distillation code](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties)
- [KD scalability paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202503271) and [official code](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties)
- [EDG paper](https://www.ijcai.org/proceedings/2025/0872.pdf), [official code/checkpoints](https://github.com/HongxinXiang/EDG), and [EDBench](https://github.com/HongxinXiang/EDBench)
- [nablaColors-3D paper](https://www.nature.com/articles/s42004-026-01944-5), [official code](https://github.com/AI4DD/nablaColors), and [Zenodo release](https://zenodo.org/records/18061300)
- [Conjugated-polymer D-MPNN paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075011/), [DOI](https://doi.org/10.3390/polym18070879), and [cited data/weights repository](https://github.com/Levitsiy/PolymersPropertiesPrediction)
- [DFT-feature-assisted optical-gap paper](https://pubs.rsc.org/en/content/articlehtml/2025/nr/d4nr03702b), [SI](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66c23a02f3f4b05290464885/original/si-file.pdf), and [MIT code/data repository](https://github.com/Liu-Group-UF/Machine-Learning-for-Accurate-Optical-Gap-Prediction-in-Conjugated-Polymers)
- [Frontier-orbital Chemprop transfer paper](https://pubmed.ncbi.nlm.nih.gov/42268043/) and [DOI](https://doi.org/10.1063/5.0333521)
- [OPoly26 paper](https://arxiv.org/pdf/2512.23117), [official OMol25 page](https://huggingface.co/facebook/OMol25), [ColabFit train record](https://materials.colabfit.org/id/DS_wfekwbgncjd3_0), [OPoly26 validation schema](https://huggingface.co/datasets/colabfit/OPoly26-val), and [fairchem code](https://github.com/facebookresearch/fairchem)
- [PubChemQC-to-conjugated-oligomer transfer paper](https://pubs.rsc.org/en/content/articlehtml/2025/me/d4me00188e) and [supporting information](https://www.rsc.org/suppdata/d4/me/d4me00188e/d4me00188e1.pdf)
- [OSCAgent paper](https://arxiv.org/html/2602.04510) and the [Harvard Clean Energy Project data record](https://www.nature.com/articles/sdata201686)
- [QuADMET-Former preprint](https://doi.org/10.26434/chemrxiv.15002429/v1), [MIT code](https://github.com/arunraja-hub/quadmetformer), [Hugging Face weights](https://huggingface.co/arunraja007/quadmetformer), and [QMugs pretraining config](https://raw.githubusercontent.com/arunraja-hub/quadmetformer/main/pretraining/configs/pretraining/qmugs.json)
- [BOA paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/718573eff1cb169316783d3e08514b5b-Paper-Conference.pdf), [official code](https://github.com/sciai-lab/boa), [README](https://raw.githubusercontent.com/sciai-lab/boa/main/README.md), and [Hugging Face checkpoint](https://huggingface.co/sciai-lab/boa)
- [C-FREE paper](https://arxiv.org/html/2509.22468), [official code](https://github.com/ariguiba/C-FREE), and [Hugging Face checkpoints](https://huggingface.co/ariguiba/C-FREE)
- [Orbformer paper](https://arxiv.org/abs/2506.19960), [OneQMC code](https://github.com/microsoft/oneqmc), [model card](https://github.com/microsoft/oneqmc/blob/main/model_card.md), and [NERD paper](https://arxiv.org/abs/2409.01306)
- [OrbitAll paper](https://arxiv.org/html/2507.03853)
- [NN-xTB paper](https://www.nature.com/articles/s41467-026-73184-z), [Code Ocean capsule](https://doi.org/10.24433/CO.8668201.v1), and [release-status repository](https://github.com/Barca-group/NN-xTB)
- [Message-Passing Delta-ML paper](https://doi.org/10.1021/acs.jctc.5c01587) and [public repository](https://github.com/AdamCoxson/Message-Passing-Delta-ML)
- [GFN2-xTB/COCONUT gap paper](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00186b), [public workflow/data](https://github.com/sthinius87/HL-gaps-pub), and [Zenodo v0.2.1 archive](https://doi.org/10.5281/zenodo.15113790)
- [QMCVNet paper record](https://authors.library.caltech.edu/records/2jygg-n1r30), [paper PDF](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/60ea947a9ab06e2e274d6cd7/original/3d-computer-vision-models-predict-dft-level-homo-lumo-gap-energies-from-force-field-optimized-geometries.pdf), and [ChemRxiv DOI](https://doi.org/10.33774/chemrxiv-2021-11r61)
- [QUED paper](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00411j), [MIT code/models](https://github.com/lmedranos/QUED), and [Zenodo archive](https://doi.org/10.5281/zenodo.17106019)
- [GLACIER paper](https://arxiv.org/html/2606.11382), [official code](https://github.com/eemokey/glacier), and [checkpoint](https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol)
- [ChemBERTa-3 paper](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00348b), [code](https://github.com/deepforestsci/chemberta3), and [Zenodo](https://zenodo.org/records/18235841)
- [ChemFM paper](https://www.nature.com/articles/s42004-025-01793-8), [code](https://github.com/TheLuoFengLab/ChemFM), and [Zenodo](https://zenodo.org/records/17450883)
- [GPSE paper](https://arxiv.org/html/2307.07107), [code](https://github.com/G-Taxonomy-Workgroup/GPSE), and [PCQM-named checkpoint](https://zenodo.org/record/8145095/files/gpse_model_pcqm4mv2_1.0.pt)
- [CondPSE](https://arxiv.org/abs/2607.25169), [GCPE publisher page](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-026-60014-0)
- [Chemprop benchmark v2](https://github.com/chemprop/chemprop_benchmark_v2) and [Zenodo data](https://zenodo.org/records/10078142)
- [ECMMR publisher page](https://www.sciencedirect.com/science/article/pii/S0957417426009103)
- [HEDMoL](https://arxiv.org/html/2602.07087), [MET](https://doi.org/10.1039/D5ME00173K), [atom-level quantum pretraining](https://link.springer.com/article/10.1186/s13321-025-00970-0), [Q-GEM](https://doi.org/10.1002/advs.202504867), [overlap-population prediction](https://doi.org/10.1093/chemle/upaf038)
- [POS-EGNN/OMol25 paper](https://pubs.rsc.org/en/Content/ArticleLanding/2026/EB/D6EB00024J), [IBM implementation](https://github.com/ibm/materials), and [Hugging Face weights/model card](https://huggingface.co/ibm-research/materials.pos-egnn)
- [AEGCNN-MTL paper](https://www.nature.com/articles/s41524-025-01917-7)
- [LUMIA paper](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00713), [MIT code](https://github.com/YajingSun-Group/LUMIA), and [Zenodo data/weights](https://zenodo.org/records/15852302)
- [MoleVers paper](https://arxiv.org/abs/2411.03537), [MPI full text](https://pure.mpg.de/rest/items/item_3632415/component/file_3632416/content), and [official code](https://github.com/ktirta/MoleVers)
- [Supervised Pretraining for Molecular Force Fields and Properties Prediction](https://arxiv.org/abs/2211.14429), [full text](https://arxiv.org/html/2211.14429), and [PubChemQC PM6 scripts/data](https://nakatamaho.riken.jp/pubchemqc.riken.jp/pm6_scripts.html)
- [PM6-ML JCTC record](https://doi.org/10.1021/acs.jctc.4c01330), [ChemRxiv PDF](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66b5fada5101a2ffa8b335b7/original/PM6-ML_preprint.pdf), and [official code/models](https://github.com/Honza-R/mopac-ml)
- [O_SMI-SSM-336M paper](https://www.nature.com/articles/s44387-025-00009-7), [IBM code](https://github.com/IBM/materials/tree/main/models/smi_ssed), and [Hugging Face weights](https://huggingface.co/ibm-research/materials.smi_ssed)
- [HieGT paper PDF](https://www.icst.pku.edu.cn/huwei/docs/20241202222445429130.pdf)
- [MolCHG paper](https://arxiv.org/html/2605.16088) and [official code](https://github.com/lhb0189/MolCHG)
- [BiScale-GTR paper](https://arxiv.org/html/2604.06336) and [official MIT repository](https://github.com/AI4Science2025/biscale-gtr-2026-tmlr)
- [FragNet paper](https://arxiv.org/html/2410.12156) and [PNNL repository](https://github.com/pnnl/FragNet)
- [MORE AAAI paper](https://ojs.aaai.org/index.php/AAAI/article/view/34262), [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/34262/36417), and [official code](https://github.com/IT-fatica/MORE)
- [FragmentNet paper](https://arxiv.org/html/2502.01184)
- [PG-MLD bioRxiv full text](https://www.biorxiv.org/content/10.64898/2026.07.29.741404v1.full) and [official code](https://github.com/lzh23399/PG-MLD)
- [ConforFormer Digital Discovery article](https://doi.org/10.1039/D6DD00096G), [indexed full article](https://www.sciencedirect.com/org/science/article/pii/S2635098X26001397), [official repository](https://github.com/EPiCs-group/ConforFormer), and [model card](https://huggingface.co/ConforFormer/ConforFormer)
- [Cardinality-Preserving Attention Channels](https://arxiv.org/html/2602.02201) and [arXiv record](https://arxiv.org/abs/2602.02201)
- [Probabilistic Contrastive Pretraining](https://arxiv.org/html/2606.11508) and [arXiv record](https://arxiv.org/abs/2606.11508)
