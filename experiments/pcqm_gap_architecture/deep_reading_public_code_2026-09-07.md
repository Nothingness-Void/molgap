# Deep Reading Batch 3: Public Codebases and Completed Pretraining Pipelines

Date: 2026-09-07

This batch reads public implementations that are close to the project’s
pretraining/teacher questions. It emphasizes what can actually be borrowed:
data manifests, geometry construction, missing-modality handling, checkpoint
layout, and ablation discipline. It does not treat a public repository or a
large foundation-model result as permission to change MolGap’s database or
ETKDG contract.

## 1. Uni-Mol2: scale laws with an ETKDG+MMFF geometry pipeline

**Primary reading.** [Uni-Mol2 paper](https://arxiv.org/html/2406.14969) and the
[Uni-Mol repository](https://github.com/deepmodeling/Uni-Mol/).

**Research question.** Uni-Mol2 studies whether molecular pretraining improves
predictably with model size, data size, and compute, rather than only changing
the architecture. It scales a two-track atom/pair Transformer from 42M to 1.1B
parameters and trains on hundreds of millions of conformations.

**Actual data and geometry.** The pretraining collection is assembled from
about 19M Uni-Mol molecules and a large ZINC20 subset, totaling about 884M
molecules/conformations and 73.7M Murcko scaffolds in the paper’s curation.
The training conformations are generated with ETKDG and optimized with RDKit
MMFF. This is unusually relevant to MolGap because the paper exposes the
construction as a reproducible data step instead of hiding it in a checkpoint.

**Model and losses.** The model has separate atom and pair streams with
shortest-path, bond, and Euclidean-distance encodings. It masks 15% of atom
tokens. For coordinate denoising it adds Gaussian coordinate noise with
standard deviation 0.2, masks atom/bond/shortest-path features with probability
0.5, aligns the noised conformer by Kabsch, and jointly predicts coordinates
and pair distances with L1 losses. The final pretraining objective is masked
atom prediction plus coordinate and distance denoising.

**Results and negative evidence.** The 1.1B model reports about 27% average
improvement over Uni-Mol on its six-task QM9 comparison. The paper also notes
that HOMO, LUMO, and Gap improvements converge as model size increases, so
larger models are not an unlimited solution for electronic targets. Training
the largest models uses 32 or 64 A100 GPUs; these numbers are evidence of
scaling behavior, not a 12-hour MolGap recipe.

**What can be borrowed.**

- explicit ETKDG/MMFF preprocessing and failure accounting;
- separate atom/pair streams with distance-denoising loss;
- Kabsch alignment before coordinate loss;
- a data/model/compute scaling ledger instead of an informal “larger is
  better” claim;
- the observation that electronic targets may hit a model-size ceiling.

**What cannot be borrowed directly.** The data pool is not the MolGap database,
the model scale is far outside the current budget, and the paper evaluates
QM9 rather than official PCQM4Mv2 Gap. External ZINC/Uni-Mol data can be an
explicit pretraining or OOD role only, never a silent augmentation.

**MolGap disposition.** **B, strongest engineering reference for ETKDG
pretraining.** It supports a possible small ETKDG-only denoising smoke test,
not a 1.1B-model proposal or a full-data database change.

## 2. VideoMol: converting PCQM conformers into a visual pretraining stream

**Primary reading.** [Nature Communications paper](https://www.nature.com/articles/s41467-024-53742-z),
[official code](https://github.com/HongxinXiang/VideoMol), and the
[public processed-data/code mirror](https://github.com/ChengF-Lab/VideoMol).

**Research question.** VideoMol asks whether a changing visual view of a
molecule can provide a general representation without hand-engineered
distances, angles, or graph features. It is a video representation method, not
a new quantum-chemistry target model.

**Actual pipeline.** The paper pretrains on 2M PCQM4Mv2 molecules and renders
60 frames per molecule, producing about 120M frames. The repository makes the
geometry-to-image path concrete: if no conformer exists, RDKit generates one
with MMFF-style optimization; PyMOL renders rotating frames; a ViT-small
backbone is used in the released example with about 21.7M parameters. The
repository describes video-level self-supervised pretraining and frame
aggregation at inference.

**Evidence.** The paper reports results across 43 drug-discovery datasets and
releases code, processed data, and trained models. The public evidence does not
establish a direct official PCQM4Mv2 Gap validation/test-dev score. It also
does not show that rasterization preserves more Gap-relevant information than
a graph/distance representation at equal compute.

**Engineering value.** VideoMol is a completed example of:

- immutable preprocessing from molecules to a large derived artifact;
- a model that aggregates multiple views at inference;
- a public checkpoint and processed-data release;
- a clear separation between pretraining data and downstream tasks.

**Risks for MolGap.** The renderer introduces view, resolution, camera, color,
and software-version dependence. Rotating images are not equivalent to
SE(3)-invariant graph geometry. The repository uses an old CUDA/PyTorch stack
and image rendering adds CPU/storage cost that is hard to justify under the
PCQM 12-hour budget.

**MolGap disposition.** **C/B for design, not a candidate.** Borrow the
preprocessing-manifest discipline and multi-view checkpoint structure if
useful. Do not replace the GraphState input with image videos unless a new
benchmark question explicitly demonstrates that rasterization adds information
not already represented by ETKDG distances/angles.

## 3. FlexMol: paired PCQM pretraining with missing-modality completion

**Primary reading.** [FlexMol paper](https://arxiv.org/abs/2510.07035),
[accepted paper copy](https://repository.kaust.edu.sa/server/api/core/bitstreams/c6a27169-de63-4108-bd09-4eb170e9ee64/content),
and [official code](https://github.com/tewiSong/FlexMol).

**Research question.** FlexMol addresses a practical problem in multimodal
pretraining: many molecules have a 2D graph but no trustworthy 3D conformer.
It tries to retain paired 2D/3D information while remaining usable when one
modality is missing.

**Actual method.** The two-stage pipeline is:

1. Stage 1 learns from paired 2D and 3D features in PCQM4Mv2 using separate 2D
   and 3D encoders, parameter sharing, contrastive alignment, and decoders for
   2D-to-3D and 3D-to-2D feature reconstruction.
2. Stage 2 continues on 2D-only or 3D-only Uni-Mol data and uses the learned
   decoder to complete the missing modality.

The repository exposes separate stage commands, Cython preprocessing, mixed
precision, gradient clipping, and single-modality fine-tuning. It also records
the official PCQM SDF MD5 (fd72bce606e7ddf36c2a832badeec6ab), which is a useful
example of an immutable data gate.

**Evidence boundary.** The paper reports improvements across molecular property
and conformation tasks and makes the missing-modality claim explicit. The
accessible paper/repository material does not provide a direct official
PCQM4Mv2 Gap score under the MolGap evaluation contract. Stage 2 uses external
Uni-Mol data, so its strongest claim is flexible multimodal pretraining, not
same-database Gap prediction.

**What can be borrowed.**

- explicit 2D/3D role-specific encoders and decoders;
- representation alignment rather than teacher property prediction;
- a missing-modality test, not only a paired-data test;
- a checksum and preprocessing record for the PCQM SDF;
- separate stage checkpoints so the contribution of each modality is
  attributable.

**Contract risks.** Stage 1 uses PCQM DFT 3D structures, while MolGap requires
ETKDG consistency. Stage 2 imports a separate pretraining corpus. A legal
MolGap adaptation would replace the 3D view with ETKDG for every role, keep
external molecules in an explicitly labeled pretraining role or remove them,
and compare 2D-only inference against the same ETKDG-only student.

**MolGap disposition.** **B, high-value teacher/missing-modality reference.**
The repository is a stronger implementation asset than a paper-only proposal,
but it does not authorize a run. The first future step would be a CPU
preprocessing/role audit, not copying the full two-stage training schedule.

## 4. MVMRL: public code with weak provenance

**Primary reading.** [MVMRL repository](https://github.com/ZangXuan/MVMRL) and the
[paper record](https://pubmed.ncbi.nlm.nih.gov/39401116/).

**Claim.** MVMRL is a self-supervised topology/geometry representation method.
Its public README says to download PCQM4Mv2 training SDF, pretrain, generate
geometry for downstream datasets, and fine-tune property tasks.

**What was verified.** The repository is public and contains separate
Pretrain and Prediction directories plus an environment file. The visible
repository has zero stars, three commits, no released checkpoint in the README,
and no direct PCQM4Mv2 Gap metric. Its code/data assumptions therefore remain
unverified at the scientific-contract level.

**MolGap disposition.** **C.** Keep it in the discovery ledger, but do not
elevate it to an experiment based on the README alone. A future code audit
would need a fixed commit, license, data checksum, pretraining objective,
geometry source, and an independently reproduced metric.

## 5. Diffusion and relation-level code assets

### 5.1 MoleculeSDE / Geom3D

The [MoleculeSDE repository](https://github.com/chao1224/MoleculeSDE) is a
small public MIT repository with a visible six-commit history, a PCQM raw-data
layout, a pretraining command, a checkpoint mapping file, and an explicit link
to [Hugging Face checkpoints](https://huggingface.co/chao1224/MoleculeSDE/tree/main).
The companion [Geom3D repository](https://github.com/chao1224/Geom3D) collects
the relevant 3D backbones and SSL scripts. This makes MoleculeSDE useful for a
read-only implementation audit: its checkpoint names identify the VE/VP and
contrastive/generative choices, and the data generator is a concrete place to
inspect how the paired conformers become graph objects.

The engineering blocker is reproducibility under the MolGap environment. The
README pins Python 3.7, PyTorch 1.9.1, PyG 2.0.2, and OGB 1.2.1; no artifact
was downloaded or executed in this audit. More importantly, the code expects
PCQM paired conformations, not an ETKDG cache. It is therefore an auditable
pretraining reference, not a drop-in checkpoint.

### 5.2 MoleBlend

The [MoleBlend repository](https://github.com/YudiZh/MoleBlend) contains the
pretraining shell, Cython build steps, the PCQM data path, a pretrained-model
link, and a four-A100 training note. It is materially more complete than a
paper-only description, but it pins a legacy PyTorch 1.7.1/PyG 1.6.3 stack and
does not expose a fixed checkpoint hash or an independently checkable PCQM Gap
split/metric in the README. The code is therefore useful for studying
relation-level blending and loss wiring, not for importing a weight into
MolGap.

### 5.3 MoleculeJAE

The NeurIPS paper and official conference record were found, but an
author-linked executable repository or checkpoint was not verified during this
audit. The implementation status is consequently lower than MoleculeSDE and
MoleBlend. Its most useful reproducibility artifact is the detailed QM9
ablation, especially the sensitivity to the contrastive coefficient.

## 6. Public-code comparison

| Project | Direct PCQM Gap evidence | Public code/assets | Geometry/data role | Safe use |
|---|---:|---:|---|---|
| Uni-Mol2 | no official PCQM Gap result in the paper | code and Uni-Mol releases | ETKDG + MMFF; huge external corpus | ETKDG denoising engineering reference |
| VideoMol | PCQM used for pretraining, no direct Gap leaderboard proof | code, processed data, models | PCQM conformer rendered as 60-frame video | manifest/checkpoint design only |
| FlexMol | PCQM paired pretraining, no direct Gap proof | code, preprocessing, stage scripts | PCQM paired DFT plus external Uni-Mol data | missing-modality teacher template |
| MoleculeSDE / Geom3D | PCQM pretraining, no direct PCQM Gap proof | MIT code, checkpoint index, Hugging Face weights, Geom3D scripts | PCQM paired conformers, old Python/PyTorch/PyG stack | data-space diffusion objective and checkpoint audit |
| MoleBlend | PCQM pretraining, no directly checkable PCQM Gap score in released table | public pretraining shell and model link | PCQM paired geometry, legacy dependency stack | relation-level pretraining wiring |
| MoleculeJAE | QM9 Gap/HOMO/LUMO evidence only | no verified official executable/checkpoint | paired PCQM geometry in paper | trajectory-loss/ablation reference |
| MVMRL | no independently checkable direct result found | code only | PCQM SDF and downstream geometry generation | discovery lead, not evidence |
| DenoiseVAE | paper appendix reports PCQM4Mv2 validation `0.0777 +/- 0.0005` with `1.44M` parameters; comparability not yet verified | public two-commit repository; no clear license/checkpoint manifest on the repository page | PCQM equilibrium 3D path; ETKDG, split, and exact config require audit | highest-priority evidence-only denoising audit |
| Pre-training via Denoising (PVD) | PCQM4Mv2 is the self-supervised upstream; QM9 table reports HOMO/LUMO/Gap improvements and the TorchMD-NET ablation isolates upstream pretraining from Noisy Nodes | MIT official repository, documented PCQM command, `denoised-pcqm4mv2.ckpt`, and QM9 fine-tuning config | DFT-equilibrium coordinates, full-PCQM upstream pool, 5-Angstrom TorchMD-NET config, and no same-contract PCQM Gap result | completed direct-PCQM denoising reference; mean-centered noise and causal controls are borrowable, but no current initialization |
| Fractional Denoising / FradNMI | PCQM4Mv2 pretraining; QM9 Frad(RN) reports `15.3/13.7/27.8` meV for HOMO/LUMO/Gap and the NMI paper reports 9/12 QM9 targets plus force/robustness transfer | MIT Frad and FradNMI repositories, Zenodo weights, Figshare source data, PCQM pretraining command, and RN/VRN configurations | Chemical-aware torsion/bond/angle noise produces `x_med`, coordinate Gaussian noise produces `x_fin`, and only `x_fin-x_med` is denoised; legacy Python/PyTorch/PyG and external DFT/RDKit+MMFF geometry | chemical-aware pretraining design reference; no checkpoint reuse or current torsion-state experiment |
| SliDe | PCQM4Mv2 label-free pretraining; 1,000-molecule DFT-force audit; QM9 HOMO/LUMO/Gap `13.6/12.3/26.2` meV; MD17 force transfer | MIT code, PCQM command, OpenFF parameter extraction, QM9/MD17 model links, and explicit BAT/random-slicing configs | DFT-equilibrium coordinates, OpenFF/Sage prior, GET/`N_v=128` cost, QM9 rather than PCQM Gap downstream | physics-informed pretraining and force-audit reference; future ETKDG-only adaptation only |
| CCMD paper | PCQM validation `0.0809` with global/local 3D-to-2D distillation and size-coordinated local loss; naive local-only loss degrades | no verified executable repository or checkpoint; primary paper equations/tables only | DFT teacher coordinates, prior Graphormer split, 68M-scale model, validation table versus abstract test-challenge claim | teacher-loss/control reference; no current score or implementation import |
| 3D-GSRD | no direct PCQM Gap score exposed | public NeurIPS 2025 code with PCQM pretraining and QM9 fine-tuning scripts | 3D masked autoencoder; README pins Python 3.8/PyTorch 2.4.1/CUDA 12.1 | selective re-mask decoder reference |
| 3D-MolT5 | PubChemQC specialist Gap `0.08` and 3D ablation `0.0791` vs `0.0968`; not official OGB PCQM evidence | Apache-2.0 code, model/data instructions, and pretraining scripts | PCQM 3D tokens plus external SELFIES/text/molecule-text corpora | compact discrete-geometry objective reference |
| MolSpectra | QM9 Gap `26.8` meV vs `31.8` meV coordinate baseline; no direct PCQM Gap score | public code and processed QM9Spectra path; license/checkpoint status needs audit | PCQM denoising plus B3LYP/def-TZVP QM9S spectra | electronic teacher/auxiliary loss reference |
| 3D-PGT | direct paper PCQM validation `0.0762` with 42.6M parameters; not a MolGap-comparable run | MIT code, PCQM pretraining command, geometry-task fusion and GPS configs | DFT 3D pretraining, old PyG 2.0.1, large model, validation-only paper protocol | completed direct-PCQM pretraining reference |
| AniDS | no direct PCQM Gap result; PCQM is label-free pretraining for MD17/OC22 | NeurIPS 2025 code, PCQM command, 4-A100 config, saved-weight path | full-covariance noise, 129M Equiformer, DFT geometry, force/energy targets | adaptive-noise engineering reference |
| 3D-EMGP | no direct PCQM Gap result; QM9/MD17 downstream only | MIT code, checkpoints, force/noise-scale tasks, `gap/homo/lumo` QM9 scripts | GEOM-QM9, Python 3.7/PyTorch 1.7/PyG 1.6.3, 3D downstream contract | physical denoising objective reference |
| Mol-MFFGE | QM9 `homo/lumo/delta` paths; no direct PCQM Gap result | official code with task-agnostic/task-aware configs and preprocessing | GEOM-QM9/SPICE/MD17, learnable noise transform, bi-level meta-learning, one visible commit | task-aware denoising reference |
| OCNet | OCELOT H-L Gap `0.008 eV`; no official PCQM result | MIT code, Zenodo LMDB datasets, checkpoints, gas-phase Gap scripts, and bimolecular assets | 10M-scale generated conjugated data, TB/DFT/xTB descriptors, dimers and films | organic-electronics teacher/OOD reference |
| LUMIA | no directly verified PCQM Gap result; paper claims strong performance on its own organic-optoelectronic tasks | MIT repository, pretraining/fine-tuning/explanation/MCTS scripts, Zenodo data archive, and retrievable pretrained dump | ~1.4M organic molecules, chemistry-informed edge/substituent masking, RGCN + contrastive objective, OCELOT/optoelectronic downstream roles | organic-domain pretraining, explanation, and artifact-contract reference; no current weights/data import |
| DFT-to-experiment frontier-orbital transfer | HOMO/LUMO transfer correlations `0.75/0.84`; not PCQM evidence | Published XGBoost/Klekota--Roth workflow, no verified project checkpoint | 11,626 DFT plus 1,198 experimental rows; theory/measurement mismatch | calibration and fragment-interpretation reference |
| OSCs_RGGN | repository claims 48,182 OSC samples; no independently checkable score | public repository only | dataset, license, split, target level, and checkpoint not exposed | quarantine discovery lead |
| GLACIER | no PCQM Gap evidence; TDC/MoleculeNet only | MIT code and Hugging Face checkpoint | 100K Enamine pretraining, graph/SMILES/RDKit descriptors, MiniMol/MolFormer teachers | frozen-embedding and dynamic multi-teacher template |
| ChemBERTa-3 | no directly verified PCQM Gap evidence in audited surfaces | MIT code, public model/config framework, Zenodo release | ZINC20/PubChem chemical foundation pretraining; optional RDKit conformer featurizer | versioned environment, benchmark, and corpus manifest |
| ChemFM | no direct PCQM Gap evidence | public code, model organization, and Zenodo record | 3B-scale causal SMILES model trained on 178M UniChem SMILES | scaling/accounting reference only |
| GPSE | PSE reconstruction/transfer evidence, no direct Gap gain | MIT code and PCQM-named Zenodo checkpoint | structural encodings, old PyG stack, provenance requires audit | frozen structural teacher or representation probe |
| CondPSE | synthetic CSL/EXP gains, no consistent molecular-property gain | paper only; no official implementation verified | polynomial graph-filter PSE pretraining | negative control for structural-pretraining claims |
| Chemprop benchmark v2 | includes `qm9_gap` and `pcqm4mv2` benchmark scripts, but is not a new MolGap claim | MIT code, Zenodo data, `data.csv`/`splits.json`, environment | Chemprop v2.0.3, explicit data/split conversion | independent loader/metric/artifact sanity package |
| [KD scalability study](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202503271) | QM9 teacher includes HOMO/LUMO/Gap; no PCQM Gap result | public SchNet/DimeNet++/TensorNet teacher/student code and downloadable models | QM9/ESOL/FreeSolv, L1+cosine latent KD, Python 3.9 | teacher-loss, capacity, and alignment-control reference |
| EDG | QM9 table includes HOMO/LUMO/Gap; no same-contract PCQM Gap result | MIT code, ImageED checkpoint, `200w_ED_feats.pkl`, ED-aware teacher checkpoint, and teacher feature artifacts | EDBench/PCQM-derived DFT electron density, structural-image teacher, QM9/rMD17 downstream, old PyTorch/PyG stack | electronic-teacher role separation and frozen-artifact manifest; no current initialization |
| nablaColors-3D / UniProp | optical absorption MAE `15.97 nm`; no PCQM Gap result | public LMDB conformers, predefined scaffold splits, correction/removal manifests, and four UniProp checkpoints with published MD5s | RDKit ETKDG2 -> MMFF94 -> GFN2-xTB -> r2SCAN-3c, with optional CPCM; low-cost-to-DFT geometry refinement and solvent embedding | geometry/artifact-contract reference only; experimental optical targets and theory/geometry roles differ from MolGap |
| DFT-feature-assisted optical-gap transfer | experimental optical-gap `MAE 0.065 eV` in the paper's own task; no PCQM Gap evidence | public MIT repository with `database-1096`, `new-database-227dp`, augmentation data, and MIT license | B3LYP-D3/6-31G* modified oligomer feature plus ECFP6; 10-fold nested CV and 227-row external validation | completed external teacher-feature and delta-vs-direct control reference; no data or feature merge |
| OPoly26 / OMol25 public polymer asset | no same-contract PCQM Gap result; paper lists HOMO/gap fields, while train metadata advertises energy/forces and the validation schema exposes `electronic_band_gap` | public paper, Hugging Face/ColabFit records, fairchem code, and stated CC-BY-4.0 release | `>6.35M` polymer DFT calculations; paper/HF say `omegaB97M-V/def2-TZVPD`, while ColabFit description says `B97M-V/def2-SVP`; condensed-phase/MD geometry is not ETKDG | manifest/schema and polymer OOD/pretraining audit only; unresolved theory/field discrepancy blocks any teacher or merge |
| PubChemQC-100K -> CO-610 transfer study | published own-task Gap MAE `0.54 eV` after transfer versus `0.71 eV` direct; no PCQM4Mv2 leaderboard evidence | public paper and ESI selection/data materials; no official code or checkpoint located | filtered same-source PubChemQC pool, external B3LYP/6-31G* oligomer target, unclear coordinate contract, and possible Track A overlap | direct-vs-transfer protocol and identity-audit reference; no data/weights import |
| GFN2-xTB/COCONUT HL-gap workflow | xTB target only; MLPR test MAE `0.210 +/- 0.001 eV`, not PCQM/B3LYP evidence | MIT GitHub `v0.2.1` plus Zenodo archive with calculated gaps/descriptors and workflow | `407k` natural products; ten RDKit conformers -> xTB/BFGS -> Boltzmann-weighted xTB gap; theory and geometry differ from MolGap | complete proxy workflow and delta-cost reference; no database, target, or teacher import |
| QMCVNet / PubChemQC PM6-to-B3LYP voxel study | same-lineage B3LYP Gap MAE `0.418--0.539 eV` from PM6/MMFF geometry; not an official PCQM4Mv2 result | Caltech paper/SI only; no audited official code, checkpoint, or reproducible data package | PubChemQC PM6/B3LYP subset, RDKit MMFF and PM6-aligned coordinates, voxel CNN, rotation augmentation; geometry is not ETKDG | negative control for low-fidelity geometry/rotation accounting; no code, rows, weights, or PM6 geometry import |
| QUED electronic-descriptor workflow | QM7-X Gap improvement with DFTB3+MBD features; no PCQM/B3LYP result | MIT repository, model pickles, HDF5 training sets, CPU scripts, and Zenodo archive; repository identifies MIT license | DFTB3+MBD global/MO/atomic descriptors plus BOB/SLATM; RDKit/MMFF -> CREST/GFN2-xTB -> lowest-energy conformer; target theory and geometry differ | strongest reusable electronic-feature packaging/ablation reference; no DFTB descriptors, rows, or weights imported |
| POS-EGNN/OMol25 electrolyte frontier workflow | small-model HOMO MAE `0.489 eV`, LUMO MAE `0.606 eV`, and all three frontier targets below `0.61 eV`; no PCQM result | IBM/materials Apache-2.0 POS-EGNN implementation, example notebook, and Hugging Face `6M` MPtrj weights; paper-specific OMol25 frontier checkpoint is not released | OMol25 `ωB97M-V/def2-TZVPD`, explicit MD solvation/ion-pair geometries, GotenNet-style equivariant encoder, HOMO/LUMO/Gap/site-charge Huber multi-task loss | physical-consistency, multi-task, and public 3D-foundation reference; no weights/data/geometry import |
| AEGCNN-MTL | QM9 HOMO `23.1 meV`, LUMO `22.0 meV`, Gap `32.1 meV`; no PCQM result | Peer-reviewed paper only; code and BDG data available from authors on request | Correlation-grouped QM9 frontier tasks, adaptive edge-aware convolution, attention, task-specific decoders, and negative-transfer control | task-grouping and multi-task ablation reference; no code/data import |

The table separates “uses PCQM” from “proves an official PCQM Gap gain.” That
distinction is mandatory because pretraining on PCQM labels or conformers can
improve a different downstream task without improving the MolGap target under
the same role and geometry rules.

## 7. Cross-code lessons worth importing into MolGap

1. **Data preparation is part of the model contract.** Uni-Mol2 and FlexMol
   expose coordinate generation, Kabsch alignment, SDF hashes, and stage
   boundaries. MolGap should keep the same level of explicitness for any future
   pretraining cache.
2. **A missing modality must be tested as missing.** A paired 2D/3D training
   result is not evidence for 2D or ETKDG-only inference. FlexMol’s 2D-only and
   3D-only stages provide the right ablation shape.
3. **Scale claims require a compute ledger.** Uni-Mol2’s 32/64-A100 training
   makes clear that foundation-model scaling is not a local architecture
   screen. Parameter count, sample count, wall time, and memory must be
   recorded together.
4. **Public code is not automatically reproducible evidence.** MVMRL shows why
   repository activity, checkpoint, data hash, license, and a numeric result
   must all be checked.
5. **Multi-view or multi-conformer inference has a cost.** VideoMol and
   Uni-Mol+ average or aggregate multiple views/conformers. A MolGap candidate
   must report total preprocessing plus inference time, not only neural-net
   forward time.
6. **Checkpoint availability is not contract compatibility.** MoleculeSDE has
   a retrievable checkpoint tree and MoleBlend has a pretrained-model link,
   but both were trained around older stacks and paired PCQM geometry. A
   checkpoint can support a provenance audit without being a legal MolGap
   initialization.
7. **Multi-teacher distillation needs an artifact contract.** GLACIER extracts
   teacher embeddings once, projects teachers independently, and gives each a
   bounded contribution. For MolGap this suggests hashing an embedding shard
   and storing the teacher revision, canonical SMILES policy, row identities,
   dtype, and shape beside it.
8. **Structural expressivity is not target evidence.** GPSE and CondPSE make
   this visible: a learned PSE can reconstruct or separate synthetic graphs yet
   fail to produce a consistent molecular-property advantage. A target-matched
   frozen control is mandatory before spending PCQM compute.
9. **A completed baseline can be more useful than another headline model.**
   Chemprop benchmark v2 exposes `data.csv`, `splits.json`, environment, and
   checkpoint/prediction output. These packaging patterns are reusable without
   replacing the current GraphState comparator or database.
10. **Electronic teachers must be separated from electronic target labels.**
    EDG shows a useful three-stage pattern: learn an electron-density
    representation, predict that representation from an accessible structural
    view, and distill it into the geometry student. The teacher can be discarded
    at inference, but the source rows, theory level, geometry, and row identities
    still need a leakage audit.
11. **A completed geometry package can be more valuable as a contract than as a
checkpoint.** nablaColors-3D publishes LMDB schemas, scaffold splits,
correction manifests, and checkpoint hashes while explicitly separating
RDKit/xTB inputs from DFT-implicit targets. That packaging pattern is
reusable for an ETKDG-only MolGap cache; its optical labels and UniProp
weights are not.
12. **A proxy feature is not automatically delta-learning.** The conjugated-
polymer optical-gap repository combines a modified-oligomer DFT gap with ECFP6
and directly predicts the experimental target. For MolGap, any future
same-database delta route must expose the low-fidelity prediction, residual
label, direct-target control, and inference cost separately.
13. **A large public database still needs a version/theory manifest.** OPoly26
shows why paper, model-card, validation-schema, and dataset-record metadata
must be compared field by field: its public surfaces do not currently agree on
the exact basis description or the split-wide frontier-field coverage. Size and
an available `electronic_band_gap` column are not enough to establish a legal
teacher for the B3LYP/6-31G*/ETKDG contract.
14. **Same-source pretraining can still be a leakage question.** The
    PubChemQC-to-CO-610 study obtains a useful transfer comparison by selecting a
    conjugation-like PubChemQC pool, but that pool is not independent of a project
    whose Track A lineage is also PubChemQC. Before reusing the idea, identity
    overlap and role assignment must be measured; target-similarity filtering is a
    protocol variable, not permission to add rows or weights.
15. **A complete proxy is still not a target match.** The GFN2-xTB/COCONUT
    workflow is valuable because it publishes its calculations, descriptors,
    conformer aggregation, and out-of-distribution theory comparison. Its
    reproducibility does not make xTB a B3LYP/6-31G* teacher, and its RDKit/xTB
    geometry path cannot be mixed with MolGap's ETKDG contract.
16. **Same-lineage low-fidelity geometry can be a useful negative control.**
    QMCVNet reports PM6/MMFF-to-B3LYP comparisons and rotation augmentation on
    a PubChemQC-derived subset, but the paper mixes coordinate contracts and
    does not publish an auditable implementation packet. Preserve the paired
    direct/proxy comparison idea, not its PM6 geometry or its direct-prediction
    numbers.
17. **Electronic-feature assets need a three-way ablation.** QUED publishes a
    geometry-only, DFTB-electronic-only, and combined descriptor comparison,
    together with field-level feature artifacts and a reproducible CPU workflow.
    This is the right control shape for a future teacher, but DFTB3+MBD
    descriptors from CREST/MMFF geometries remain a different theory and input
    contract from MolGap.
18. **A physical relation can define task weights, not just an extra label.**
    POS-EGNN makes HOMO and LUMO primary predictions, uses Gap as a lower-weight
    consistency term, and adds a site-charge head. AEGCNN-MTL independently
    shows that the benefit of multi-task learning follows measured task
    relatedness and can become negative for weakly coupled properties. Any future
    MolGap multi-task screen therefore needs a no-auxiliary control, explicit
    `Gap - (LUMO - HOMO)` diagnostics, and a negative-transfer report.
19. **Domain knowledge can be encoded as a pretraining view, but must be
    versioned as data.** LUMIA's masked-edge/substituent positives and MCTS
    explanation path are useful only when the transformation, source rows,
    checkpoint hash, and downstream folds are frozen. A chemically plausible
    augmentation is not automatically label-free or ETKDG-compatible.
20. **Upstream compatibility is an empirical gate, not a size assumption.**
    PVD improves QM9 across tested downstream sizes and benefits from more
    upstream PCQM structures until saturation, but PCQM-to-OC20 transfer does
    not improve final validation. A MolGap pretraining claim therefore needs
    an upstream-size curve, a same-contract random-init control, and an
    explicit molecular-distribution compatibility check.
21. **Denoising must be separated into three causal components.** PVD's
    TorchMD-NET table compares random initialization, downstream Noisy Nodes,
    and PCQM pretraining. This is the right control shape for any future
    ETKDG audit; a single pretraining-vs-baseline comparison cannot tell
    whether the gain came from the denoising head, the upstream representation,
    or the changed optimization path.
22. **Chemical-aware noise changes the sampling distribution, not the target
    contract.** Frad's theorem permits arbitrary CAN only because the model
    regresses the final isotropic coordinate component. A future ETKDG audit
    must retain the intermediate `x_med` and final `x_fin` manifests; adding
    torsion noise and regressing the whole hybrid displacement would be a
    different objective with a different force interpretation.
23. **A robustness result is not geometry equivalence.** Frad remains useful
    after RDKit Distance Geometry+MMFF pretraining conformers become less
    accurate, but the paper still reports larger errors and does not test the
    MolGap ETKDG contract. The result supports a robustness control, not a
    license to mix RDKit/MMFF, DFT, and ETKDG coordinates.
24. **Physics-informed denoising has an estimator contract.** SliDe's BAT
    prior, random slicing, finite-difference scale, sample count, ring rules,
    and force audit are one coupled objective. A future ETKDG adaptation must
    freeze all of them and compare against coordinate denoising; copying only
    bond/angle/torsion noise would not reproduce the paper's force target.
25. **Global and local distillation must be separated.** CCMD's PCQM ablation
    shows that pooled/global alignment helps slightly, naive atom-level
    alignment hurts, and size-coordinated all-layer alignment recovers the
    gain. A future teacher experiment needs global-only, local-only, and
    coordinated controls; a single combined KD loss is not an attribution.

## 8. Evidence-only next actions

- For Uni-Mol2, record the exact ETKDG/MMFF implementation and failure behavior;
  do not copy its external dataset or model scale.
- For FlexMol, inspect the stage-1 loss and the 2D-only inference path; check
  whether the decoder can be isolated as a teacher representation target.
- For VideoMol, keep the repository as a reproducible-artifact reference but
  do not spend PCQM budget on rasterization without a representation ablation.
- For MVMRL, wait for a fixed revision and independently checkable result
  before considering it evidence.
- For MoleculeSDE and MoleBlend, record the exact repository revision and
  checkpoint hash before any local smoke test; do not use their PCQM geometry
  as an ETKDG cache.
- For DenoiseVAE, verify the reported PCQM appendix number against the exact
  split, coordinate source, checkpoint, license, and inference path before
  treating it as a candidate; do not paste its representation into GraphState.
- For 3D-GSRD and 3D-MolT5, borrow only the selective decoder or discrete local
  geometry objective after architecture selection; do not import their 3D or
  text corpora into the current database.
- For MolSpectra, treat QM9Spectra as an external-theory teacher reference and
  audit B3LYP/def-TZVP, identity overlap, and license before any teacher query.
- For 3D-PGT, preserve the `0.0762` as paper validation evidence only; do not
  compare it with the bounded GraphState screen without an exact role and
  geometry audit.
- For AniDS and 3D-EMGP, borrow noise/objective structure only; their force,
  QM9, MD17, and OC22 results are not PCQM Gap evidence.
- For OCNet, freeze the Zenodo release IDs and hashes before any teacher/OOD
  audit; do not treat its OCELOT Gap or TB labels as PCQM labels.
- For OSCs_RGGN, require a primary paper and dataset manifest before treating
  the 48K claim as evidence.
- For GLACIER, borrow only the frozen-embedding and multi-teacher ablation
  pattern after architecture selection; do not import Enamine, TDC, or teacher
  labels into the current database.
- For GPSE/CondPSE, require a checkpoint lineage and a same-contract Gap control;
  synthetic PSE gains are not sufficient evidence.
- For Chemprop benchmark v2, use only as an independent loader/metric/artifact
  sanity reference if a future audit explicitly calls for it.
- For the 2025 KD scalability study, borrow the teacher/no-teacher and student-
  capacity control matrix plus embedding-alignment diagnostic; do not treat QM9
  teacher numbers as PCQM Gap evidence.
- For EDG, audit the released feature index, checkpoint hashes, license, and
  PCQM4Mv2 row overlap before considering any electronic teacher. If revisited,
  rerender the teacher input from the exact ETKDG conformer contract and compare
  against a frozen-teacher/no-teacher control; do not import the 2M ED pool or
  its QM9 numbers into MolGap.
- For nablaColors-3D/UniProp, borrow only the versioned artifact pattern:
  correction/removal manifests, split files, LMDB schema, and checkpoint hashes.
  Do not import its optical data, solvent encoder, or weights; its geometry
  refinement route is a separate contract from MolGap's frozen ETKDG path.
- For the conjugated-polymer D-MPNN paper, retain the domain-matched
  pretraining ablation but do not treat the unresolved cited repository as a
  reusable artifact; its polymer experimental labels and MMFF94s/TD-DFT
  geometry are external to MolGap.
- For the UF optical-gap repository, borrow only the distinction between a
  DFT teacher feature and a strict residual delta target, plus its nested-CV
  and group-based extrapolation reporting. Do not import its optical-gap rows
  or calculate its feature on the current PCQM track without a new contract.
- For OPoly26, first freeze the paper/Hugging Face/ColabFit/fairchem revisions,
  reconcile functional/basis and frontier-field coverage, inspect the actual
  split files and hashes, and quantify identity overlap. Until that audit is
  complete, treat it only as an external polymer database/packaging reference;
  do not import rows, frontier fields, or weights into Track A/B.
- For the PubChemQC-100K/CO-610 transfer study, retrieve and hash the ESI,
  reconstruct the `>6 double bonds`/`gap < 6 eV` selection rule, canonicalize
  every identity against Track A and PCQM roles, and verify the oligomer
  coordinate method. Until then, keep the frozen-layer plus one-block recipe as
  a literature control only; do not copy its filtered pool or checkpoint.
- For the GFN2-xTB/COCONUT workflow, freeze GitHub/Zenodo `v0.2.1`, record the
  MIT and upstream COCONUT terms separately, inspect the actual descriptor/data
  schema, and compare its ten-conformer xTB/Boltzmann contract with an
  ETKDG-only single-proxy audit. Never report its xTB MAE as PCQM evidence.
- For QMCVNet, freeze the Caltech paper/SI hashes and verify overlap between its
  PubChemQC PM6/B3LYP subset and Track A. Retain only its explicit
  low-fidelity/rotation-control framing; do not reuse PM6/MMFF coordinates or
  treat direct high-level regression as strict delta-learning.
- For QUED, freeze the Digital Discovery paper, MIT repository, and Zenodo
  release; inspect the D_QM schema, model pickles, HDF5 training sets, and
  license terms. Borrow only the geometry/electronic/combined ablation shape;
  do not calculate DFTB/CREST features on the current database without a new
  ETKDG-compatible teacher contract.
- For POS-EGNN, freeze the paper, IBM repository, and Hugging Face model card
  separately. The public checkpoint is an MPtrj energy/force/stress model, not
  the paper's OMol25 frontier-orbital model; inspect it only as an external
  3D-foundation implementation reference.
- For AEGCNN-MTL, record only its QM9 correlation grouping and within-architecture
  single-task/multi-task comparison. Its unpublished BDG dataset and absent
  public code do not clear a reproduction or current experiment.
- For LUMIA, freeze the ACS paper, GitHub revision, and Zenodo MD5s; inspect the
  exact pretraining CSV columns and the OCELOT fold files before any external
  teacher audit. Borrow only the knowledge-mask definition and explanation
  artifact pattern; do not import its RGCN checkpoint or optoelectronic rows.
- For PVD, use the paper and official `ET-PCQM4MV2.yaml` as the baseline
  denoising specification. If a future ETKDG audit is authorized, reproduce
  the mean-centered vector target and separately compare random-init,
  downstream denoising-only, and upstream-pretrained controls. Do not use the
  released DFT-coordinate checkpoint or its QM9 numbers as MolGap evidence.
- For Frad/FradNMI, freeze the ICML/NMI paper versions, repository revisions,
  Zenodo weight record, source-data record, and the RN/VRN scales. If a future
  ETKDG-only pretraining audit is opened, preserve `x_eq`, `x_med`, and
  `x_fin`, regress only the final coordinate-noise component, and compare
  against coordinate denoising and no pretraining. Do not reuse DFT or
  RDKit+MMFF conformers, the external checkpoint, or the QM9 frontier scores.
- For SliDe, inspect and freeze the official OpenFF parameter extraction,
  ring/degree handling, `N_v`/`sigma` finite-difference policy, and GET loss
  before any ETKDG-only adaptation. Use the paper's force audit only as a
  train-role diagnostic; do not import QM9 numbers or SliDe weights.
- For CCMD, preserve the global-token/local-atom split and its size-normalized
  ablations as a teacher protocol reference. No executable or checkpoint was
  verified, so do not claim reproducibility or copy the `0.0809` validation
  number into the current benchmark.
- Any future use of PCQM SDF must verify the official checksum and ensure that
  validation/test roles remain sealed.

## Primary-source index

- [Uni-Mol2 paper](https://arxiv.org/html/2406.14969),
  [Uni-Mol code](https://github.com/deepmodeling/Uni-Mol/)
- [VideoMol paper](https://www.nature.com/articles/s41467-024-53742-z),
  [code](https://github.com/HongxinXiang/VideoMol)
- [FlexMol paper](https://arxiv.org/abs/2510.07035),
  [code](https://github.com/tewiSong/FlexMol)
- [MVMRL code](https://github.com/ZangXuan/MVMRL),
  [paper record](https://pubmed.ncbi.nlm.nih.gov/39401116/)
- [MoleculeSDE](https://github.com/chao1224/MoleculeSDE),
  [Geom3D](https://github.com/chao1224/Geom3D), and
  [checkpoints](https://huggingface.co/chao1224/MoleculeSDE/tree/main)
- [MoleBlend](https://github.com/YudiZh/MoleBlend)
- [MoleculeJAE paper](https://arxiv.org/html/2312.03475)
- [DenoiseVAE paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html),
  [paper PDF](https://openreview.net/attachment?id=ym7pr83XQr&name=pdf), and
  [code](https://github.com/liuyurou1/DenoiseVAE)
- [EDG paper](https://www.ijcai.org/proceedings/2025/0872.pdf),
  [code and checkpoints](https://github.com/HongxinXiang/EDG), and
  [EDBench](https://github.com/HongxinXiang/EDBench)
- [nablaColors-3D paper](https://www.nature.com/articles/s42004-026-01944-5),
  [code](https://github.com/AI4DD/nablaColors), and
  [Zenodo release](https://zenodo.org/records/18061300)
- [3D-GSRD paper](https://arxiv.org/abs/2510.16780) and
  [code](https://github.com/WuChang0124/3D-GSRD)
- [3D-MolT5 paper](https://arxiv.org/html/2406.05797) and
  [code](https://github.com/QizhiPei/3D-MolT5)
- [MolSpectra paper](https://arxiv.org/html/2502.16284) and
  [code](https://github.com/AzureLeon1/MolSpectra)
- [3D-PGT paper](https://arxiv.org/html/2306.07812) and
  [code](https://github.com/LARS-research/3D-PGT)
- [AniDS paper](https://arxiv.org/html/2510.22123) and
  [code](https://github.com/ZeroKnighting/AniDS)
- [3D-EMGP paper](https://ojs.aaai.org/index.php/AAAI/article/view/25978) and
  [code](https://github.com/jiaor17/3D-EMGP)
- [Mol-MFFGE paper](https://www.sciencedirect.com/science/article/pii/S0031320325001918) and
  [code](https://github.com/Yufei-Luo/Mol-MFFGE)
- [PVD paper](https://arxiv.org/html/2206.00133), [official code/checkpoint](https://github.com/shehzaidi/pre-training-via-denoising), and
  [PCQM configuration](https://github.com/shehzaidi/pre-training-via-denoising/blob/main/examples/ET-PCQM4MV2.yaml)
- [Frad ICML paper](https://arxiv.org/html/2307.10683), [Frad code](https://github.com/fengshikun/Frad), [Frad NMI paper](https://arxiv.org/html/2407.11086), [FradNMI code](https://github.com/fengshikun/FradNMI), [Zenodo weights](https://zenodo.org/records/12697467), and [source data](https://doi.org/10.6084/m9.figshare.25902679.v1)
- [SliDe paper](https://arxiv.org/html/2311.02124), [official code](https://github.com/fengshikun/SliDe), and [ICLR record](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a1d69d1f64c6b6df105b15984ca527a-Abstract-Conference.html)
- [CCMD paper](https://arxiv.org/html/2211.16712); no verified executable repository or checkpoint was found in this audit
- [OCNet paper](https://www.nature.com/articles/s41524-025-01788-y),
  [code](https://github.com/545487677/OCNet), and
  [Zenodo bimolecular assets](https://zenodo.org/records/14934728)
- [DFT-to-experiment frontier-orbital transfer paper](https://www.nature.com/articles/s41524-024-01403-6)
- [OSCs_RGGN code lead](https://github.com/AsadKhanJBNU/OSCs_RGGN)
- [Conjugated-polymer D-MPNN paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075011/),
  [cited repository](https://github.com/Levitsiy/PolymersPropertiesPrediction)
- [DFT-feature-assisted optical-gap paper](https://pubs.rsc.org/en/content/articlehtml/2025/nr/d4nr03702b),
  [SI](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66c23a02f3f4b05290464885/original/si-file.pdf),
  and [MIT code/data repository](https://github.com/Liu-Group-UF/Machine-Learning-for-Accurate-Optical-Gap-Prediction-in-Conjugated-Polymers)
- [OPoly26 paper](https://arxiv.org/pdf/2512.23117), [official OMol25 page](https://huggingface.co/facebook/OMol25), [ColabFit train record](https://materials.colabfit.org/id/DS_wfekwbgncjd3_0), [OPoly26 validation schema](https://huggingface.co/datasets/colabfit/OPoly26-val), and [fairchem code](https://github.com/facebookresearch/fairchem)
- [PubChemQC-to-conjugated-oligomer transfer paper](https://pubs.rsc.org/en/content/articlehtml/2025/me/d4me00188e) and [supporting information](https://www.rsc.org/suppdata/d4/me/d4me00188e/d4me00188e1.pdf)
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
  [Hugging Face weights/model card](https://huggingface.co/ibm-research/materials.pos-egnn)
- [AEGCNN-MTL paper](https://www.nature.com/articles/s41524-025-01917-7)
- [LUMIA paper](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00713), [MIT code](https://github.com/YajingSun-Group/LUMIA), and [Zenodo data/weights](https://zenodo.org/records/15852302)
- [GLACIER paper](https://arxiv.org/html/2606.11382), [official code](https://github.com/eemokey/glacier), and [checkpoint](https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol)
- [ChemBERTa-3 paper](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00348b), [code](https://github.com/deepforestsci/chemberta3), and [Zenodo](https://zenodo.org/records/18235841)
- [ChemFM paper](https://www.nature.com/articles/s42004-025-01793-8), [code](https://github.com/TheLuoFengLab/ChemFM), and [Zenodo](https://zenodo.org/records/17450883)
- [GPSE paper](https://arxiv.org/html/2307.07107), [code](https://github.com/G-Taxonomy-Workgroup/GPSE), and [PCQM-named checkpoint](https://zenodo.org/record/8145095/files/gpse_model_pcqm4mv2_1.0.pt)
- [CondPSE](https://arxiv.org/abs/2607.25169), [GCPE publisher page](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-026-60014-0)
- [Chemprop benchmark v2](https://github.com/chemprop/chemprop_benchmark_v2) and [Zenodo data](https://zenodo.org/records/10078142)
- [ECMMR publisher page](https://www.sciencedirect.com/science/article/pii/S0957417426009103)
- [KD scalability paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202503271) and [official code](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties)
