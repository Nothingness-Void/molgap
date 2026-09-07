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
3. **Adaptive-noise pretraining.** DenoiseVAE is the strongest new
   direct-PCQM validation lead: its ICLR 2025 appendix reports `0.0777 +/-
   0.0005` on PCQM4Mv2 with `1.44M` parameters and a public repository. The
   exact split, coordinate path, checkpoint, license, and ETKDG compatibility
   remain open, so this is an audit lead rather than a permitted initialization.
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
[public repository](https://github.com/liuyurou1/DenoiseVAE). This is the
strongest direct-PCQM number found in the continuation, but it is not yet a
MolGap-comparable result: the exact split, equilibrium-coordinate pipeline,
checkpoint, inference inputs, and repository license still require an audit.

The method is therefore a high-priority evidence-only lead, not a checkpoint
initialization. A legal adaptation would learn molecule-adaptive noise from
ETKDG-generated train-role coordinates and use ETKDG again for downstream
inference, with a fresh random-init GraphState control. The paper's reported
`sigma` and KL settings are starting points for an audit, not values to copy
without measuring the MolGap coordinate distribution.

**Disposition: B.** Add DenoiseVAE to the smoke/audit queue only after the
repository revision, license, split, coordinate, and checkpoint gates are
closed. Do not use its PCQM number as a leaderboard result or paste its
representation into the architecture screen.

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

### 4.2 DGT — useful dual-graph idea, no direct MolGap admission

[Dual Graph Transformer](https://github.com/zhangsy-ryan/DGT) provides public
code for separate atom and bond graphs with cross-level interactions and
optional 3D descriptors; the related [paper](https://www.nature.com/articles/s41467-026-75005-9)
reports ablations on QM9. The bond graph is conceptually close to the
project’s persistent EdgeState, and the existing GraphState anchor already
contains a bond-level state. Reimplementing DGT wholesale would therefore be
mostly a large architecture transplant, not a clean new information-flow
question. The repository does not establish a direct, ETKDG-compatible PCQM
Gap gain for the MolGap contract.

**Disposition: B for design reference, not a new queue item.** A specific
bond--atom interaction could be proposed only with an explicit new hypothesis
that is not already covered by EdgeState/GraphState evidence.

### 4.3 Weak or non-comparable public claims

The public repository
[spectral-temporal-curriculum-molecular-gaps](https://github.com/A-SHOJAEI/spectral-temporal-curriculum-molecular-gaps)
claims PCQM training and reports `0.268 eV`, but has zero stars, three visible
commits, no paper or independently traceable run provenance, and a result far
below the established PCQM references. Its README is not sufficient evidence
for a candidate and it is excluded.

The same rule applies to any recent paper that reports validation-as-test
numbers, a single unseeded run, an external geometry source, or a stale
leaderboard comparison without a reproducible artifact. Search-result snippets
and repository README claims are discovery signals, not acceptance evidence.

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

M1--M4 each require a separate dated protocol. They cannot be stacked into one
large screen: the result would not identify whether denoising, distillation,
the proxy, or the auxiliary label caused a gain.

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
| DenoiseVAE | B | Yes, audit lead before any protocol | Direct PCQM4Mv2 validation number and public code, but split, checkpoint, license, and ETKDG contract remain open. |
| Pre-training via Denoising (PVD) | A/B | No current experiment; future ETKDG protocol reference | Direct PCQM4Mv2 self-supervised upstream, MIT code/checkpoint, QM9 HOMO/LUMO/Gap transfer, and explicit random-init/Noisy-Nodes/pretraining controls; DFT coordinates and no same-contract PCQM Gap result block current initialization. |
| Fractional Denoising (Frad/FradNMI) | A/B | No current experiment; future chemical-aware ETKDG pretraining reference | Public ICML/NMI papers, MIT code/weights/data, RN/VRN chemical-aware noise, and QM9/force/robustness ablations; DFT/RDKit+MMFF geometry, QM9 downstream, legacy stack, and closed torsion route block current initialization. |
| SliDe | A/B | No current experiment; future force-consistent ETKDG pretraining reference | Public ICLR paper and MIT code/checkpoints; BAT bond/angle/torsion prior, random slicing, PCQM label-free pretraining, force audit, and QM9 frontier ablations are unusually complete. DFT coordinates, OpenFF/Sage prior, QM9 rather than PCQM Gap downstream, and estimator/model cost block direct initialization. |
| CCMD | B | No current experiment; future teacher-loss control template | Direct PCQM validation shows global molecular-token distillation can help, while naive local atom-token loss hurts and size-coordinated local loss repairs it. DFT teacher geometry, non-audited split wording, large model scale, and absent verified code/checkpoint require a separate ETKDG contract. |
| 3D-PGT | B | Historical direct-PCQM reference; conditional future contract only | `0.0762` validation evidence and public MIT code, but 42.6M DFT-pretrained GPS does not fit the bounded ETKDG screen. |
| AniDS | B | Design reference only | Public PCQM label-free pretraining and anisotropic noise, but 129M force-field model, DFT geometry, and no direct PCQM Gap result. |
| 3D-EMGP | B | Design reference only | Public physical denoising objective and QM9 orbital fine-tuning, but GEOM-QM9/MD17 rather than PCQM. |
| Mol-MFFGE | B | Task-aware denoising reference only | Public paper/code and QM9 `homo/lumo/delta` path, but GEOM/SPICE/MD17 geometry and no direct PCQM Gap result. |
| OCNet | B | Yes, external teacher/OOD route after identity audit | Strong conjugated-domain paper/code/data/weight package, but theory, geometry, dimers, and labels differ from PCQM. |
| nablaColors-3D / UniProp | B | Geometry/artifact reference only | Peer-reviewed optical benchmark with public LMDB/splits/manifests/checkpoint hashes and explicit low-cost-to-DFT refinement; experimental targets, solvent/theory roles, and ETKDG2 geometry differ from MolGap. |
| Conjugated-polymer D-MPNN pretraining | B | External domain-matched pretraining reference only | Published three-way transfer comparison is relevant, but targets and geometries are external; cited artifact URL currently returns `404` in the audit. |
| DFT-feature-assisted optical-gap transfer | B | Delta/teacher protocol reference only | Public MIT code/data and full paper/SI are available, but it predicts experimental optical gaps from an oligomer DFT feature rather than a same-label residual. |
| Frontier-orbital Chemprop transfer | C | No current experiment | Published abstract/preview reports a promising GFN2-xTB-trimer transfer pattern, but the full reproducibility packet is not available in the audited sources. |
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
| DGT transplant | B | No as a queue item | Public code, but no direct matching PCQM Gap evidence and much of the idea overlaps EdgeState. |
| Weak README-only PCQM projects | C | No | No independent provenance or comparable evidence. |

No remote job, external dataset merge, pretrained initialization, or production
change was made while writing this audit.

## Primary sources

- [OGB PCQM4Mv2 specification](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
- [Self-Conditioned Denoising paper](https://arxiv.org/html/2603.17196v1), [official code](https://github.com/TyJPerez/SelfConditionedDenoisingAtoms), [PCQ checkpoint](https://huggingface.co/Ty-Perez/ct-scd-pcq)
- [DenoiseVAE paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html), [paper PDF](https://openreview.net/attachment?id=ym7pr83XQr&name=pdf), and [code](https://github.com/liuyurou1/DenoiseVAE)
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
- [ViSNet PCQM technical report](https://ogb.stanford.edu/paper/neurips2022/pcqm4mv2_ViSNet.pdf)
- [DelFTa code](https://github.com/josejimenezluna/delfta), [DelFTa documentation](https://delfta.readthedocs.io/en/latest/), [delta-QML paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9093086/)
- [GW/DFT delta-learning paper](https://doi.org/10.1088/2632-2153/acf545)
- [DeMol paper](https://arxiv.org/html/2603.00568), [author-linked repository](https://github.com/LiuYunqing/DeMol)
- [DGT code](https://github.com/zhangsy-ryan/DGT), [DGT paper](https://www.nature.com/articles/s41467-026-75005-9)
- [Generic molecular knowledge-distillation code](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties)
- [KD scalability paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202503271) and [official code](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties)
- [EDG paper](https://www.ijcai.org/proceedings/2025/0872.pdf), [official code/checkpoints](https://github.com/HongxinXiang/EDG), and [EDBench](https://github.com/HongxinXiang/EDBench)
- [nablaColors-3D paper](https://www.nature.com/articles/s42004-026-01944-5), [official code](https://github.com/AI4DD/nablaColors), and [Zenodo release](https://zenodo.org/records/18061300)
- [Conjugated-polymer D-MPNN paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075011/), [DOI](https://doi.org/10.3390/polym18070879), and [cited data/weights repository](https://github.com/Levitsiy/PolymersPropertiesPrediction)
- [DFT-feature-assisted optical-gap paper](https://pubs.rsc.org/en/content/articlehtml/2025/nr/d4nr03702b), [SI](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66c23a02f3f4b05290464885/original/si-file.pdf), and [MIT code/data repository](https://github.com/Liu-Group-UF/Machine-Learning-for-Accurate-Optical-Gap-Prediction-in-Conjugated-Polymers)
- [Frontier-orbital Chemprop transfer paper](https://pubmed.ncbi.nlm.nih.gov/42268043/) and [DOI](https://doi.org/10.1063/5.0333521)
- [OPoly26 paper](https://arxiv.org/pdf/2512.23117), [official OMol25 page](https://huggingface.co/facebook/OMol25), [ColabFit train record](https://materials.colabfit.org/id/DS_wfekwbgncjd3_0), [OPoly26 validation schema](https://huggingface.co/datasets/colabfit/OPoly26-val), and [fairchem code](https://github.com/facebookresearch/fairchem)
- [PubChemQC-to-conjugated-oligomer transfer paper](https://pubs.rsc.org/en/content/articlehtml/2025/me/d4me00188e) and [supporting information](https://www.rsc.org/suppdata/d4/me/d4me00188e/d4me00188e1.pdf)
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
