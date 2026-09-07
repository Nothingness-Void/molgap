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

### 5.4 NVIDIA PCQM4Mv2 winner: completed heterogeneous ensemble and OOF stacking

**Primary reading.** [Heterogenous Ensemble of Models for Molecular Property
Prediction](https://arxiv.org/abs/2211.11035) and the authors' [MIT-licensed
NVIDIA-PCQM4Mv2 repository](https://github.com/jfpuget/NVIDIA-PCQM4Mv2).

This is a completed PCQM4Mv2 artifact rather than a new pretraining paper. The
paper reports a test-challenge MAE of `0.0723`, a validation MAE of `0.07145`,
and less than two hours for inference. The released repository contains the
data-fold builder, Transformer-M/Graphormer, molecular Transformer, PD-DGN,
CNN, and ensemble directories, with Docker and per-model run instructions.
The repository page exposes the MIT license and instructs users to build folds,
train the base models, and run the ensemble notebook.

The important method is not simply “many models.” The authors first diagnose a
distribution shift: their train split contains molecules with at most 20 heavy
atoms, while validation, test-dev, and test-challenge include molecules up to
51 heavy atoms. They combine the official train and validation rows, construct
24 folds, use four folds as validation in the reported screen, train each base
model on the other 23 folds, and save out-of-fold (OOF) predictions. A
HuberRegressor is then fit on OOF predictions and applied to held-out test
predictions. This makes the stacking target auditable, but it is not compatible
with treating official validation/test-dev as sealed during MolGap model
selection.

The base-model ablations are useful even when the final ensemble is too large:

- Transformer-M samples 2D-only, 3D-only, and 2D+3D channels. The documented
  large configuration uses an 18-layer, width-768 model, position noise `0.2`,
  batch size `128`, 454 epochs, and eight distributed processes.
- A denoising variant adds an atom-position head and an auxiliary denoising
  loss. A separate KPGT-style variant predicts a 512-bit fingerprint and about
  200 descriptors through auxiliary heads; the paper reports an initial `0.007`
  MAE improvement over its baseline, then trains `lambda=0.1` and `0.2`
  variants.
- The published Huber weights assign a negative coefficient to the denoising
  base model (`about -0.03` to `-0.04`) while retaining positive weights for
  several weaker 2D/PD-DGN/image models. This is evidence that an auxiliary
  task can be useful for diversity without being individually strong, but it
  is not evidence that denoising improves a single MolGap model.

**What transfers to MolGap.** The highest-value lessons are a molecule-size
shift diagnostic, explicit OOF prediction manifests, robust linear stacking
after candidate training, and a separate ablation for denoising versus
descriptor/fingerprint regularization. A small delivery-time ensemble of
already accepted GraphState candidates could borrow the Huber/OOF protocol
without changing the database or model inputs.

**Hard limits.** The competition solution uses provided PCQM 3D SDF geometry,
RDKit-derived image views, train+validation rows for its fold construction,
large 18-layer models, and many checkpoints. It is therefore not an ETKDG
single-model comparison, not a valid replacement for the current official-role
evaluation, and not a bounded architecture-discovery screen. No checkpoint,
image artifact, fold file, or PCQM row is imported.

**MolGap disposition.** **A for completed code and direct PCQM evidence; B for
current scientific transfer.** Keep it as a finished delivery/OOF-stacking
reference. Do not compare its `0.0723` challenge result with the current
GraphState seed screen without matching roles, geometry, and candidate count.

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
| GraphQPT | PCQM HLG is a pretraining source; no direct PCQM Gap improvement is reported | MIT code, analysis notebooks, YAML configs, and curated pretraining datasets linked through Zenodo | 2D Graphormer pretraining; external atom-level data use MMFF94s -> GFN2-xTB -> B3LYP/def2-SVP geometry, while PCQM contributes a graph-level HLG label | electronic-teacher and representation-analysis reference; no current labels or weights |
| DGT | no direct official PCQM Gap leaderboard score; paper reports PCQM Gap pretraining followed by QM9 HOMO/LUMO MAE `0.0306 eV` and LUMO MAE `0.0240 eV` at 10K | MIT repo, MIT license, Zenodo code package, Figshare source data, and visible QM9 configs | 10K/100K/1M PCQM Gap pretraining; QM9 B3LYP/6-31G(2df,p), DFT/MMFF/UFF 3D comparison; visible repo exposes QM9 configs only | pretraining and dual atom/bond artifact reference; no current initialization |
| QuantumCanvas | paper reports gap `0.201 +/- 0.020 eV` for GATv2 and QM9 transfer; no direct PCQM Gap result | MIT code, CC-BY Zenodo dataset, `dataset_combined.npz`, MD5, loaders, and `REPRODUCE.md` | 2,850 diatomic element pairs, SCC-DFTB/DFTB+ PTBP labels, ten-channel orbital/charge images, explicit coordinates, composition-held-out splits | two-body electronic-teacher and manifest reference; no current data, image features, or weights |
| TMC-Delta-ML | peer-reviewed paper reports direct-versus-Delta gains and lower-cost fidelity tradeoffs; target is tmQMg, not PCQM | MIT code, MIT license, Zenodo low-fidelity graph archive, tmQMg auto-download, and fixed CSD-ID split files | GFN2-xTB geometries with LSDA/LANL2DZ or PBE0-D3BJ/def2-TZVP low-fidelity u-NatQG graphs; HOMO--LUMO path is explicit | completed Delta-learning/cost protocol reference; no current labels, features, or initialization |
| SelectedML | QM7/QM9 class-conditioned QML learning curves and direct/Delta-QML scripts; no PCQM result | public repository with functional-group frequency analysis, CM/BoB/SLATM preparation, CV/learning-curve scripts, and three-seed controls | KRR on QM7/QM9 at GW/B3LYP/ZINDO roles; structural classes are not an ETKDG/PCQM contract | chemical-family stratification and mixture-of-experts diagnostic; no current data, weights, or split change |
| HLP-Stack | near-perfect QM9 HOMO/LUMO stacking claims; no PCQM result | public paper/repo with raw and processed data, notebooks, saved models, figures, and environment | RDKit 2D plus DFT-derived QM9 3D descriptors at B3LYP/6-31G(2df,p); target-adjacent feature path and no ETKDG contract | negative descriptor/leakage audit; no model or quantum-feature import |
| PCQM 3D-prior distillation repo | documented result is a synthetic smoke, not a PCQM leaderboard result | one-commit codebase, split/geometry manifest fields, checkpoint/report output paths, and paired-bootstrap smoke metrics | intended DFT or RDKit ETKDGv3+UFF teacher; actual artifact uses synthetic data and the reported student loses to both baseline controls | negative distillation control and manifest template; no current experiment |
| DenoiseVAE | paper appendix reports PCQM4Mv2 validation `0.0777 +/- 0.0005` with `1.44M` parameters; code is not directly runnable as published | public two-commit repository; no clear license/checkpoint manifest; default config points to GEOM; public model has an undefined `noneed` return | PCQM builder uses RDKit `EmbedMolecule` + MMFF and stores a DFT-aligned train position; not the ETKDG contract | paper/method audit only; code C; no initialization |
| Self-Conditioned Denoising | PCQ pretraining and QM9 frontier transfer controls; official PCQM pretraining command and public `ct-scd-pcq` checkpoint | MIT/GitHub implementation, Hugging Face checkpoint, YAML/config surface, and two-pass conditional denoising | Equilibrium PCQM coordinates, TorchMD-Net-style 3D path, no direct PCQM Gap improvement, and no ETKDG equivalence | Highest-confidence implementation-backed pretraining audit; no checkpoint import |
| Pre-training via Denoising (PVD) | PCQM4Mv2 is the self-supervised upstream; QM9 table reports HOMO/LUMO/Gap improvements and the TorchMD-NET ablation isolates upstream pretraining from Noisy Nodes | MIT official repository, documented PCQM command, `denoised-pcqm4mv2.ckpt`, and QM9 fine-tuning config | DFT-equilibrium coordinates, full-PCQM upstream pool, 5-Angstrom TorchMD-NET config, and no same-contract PCQM Gap result | completed direct-PCQM denoising reference; mean-centered noise and causal controls are borrowable, but no current initialization |
| Fractional Denoising / FradNMI | PCQM4Mv2 pretraining; QM9 Frad(RN) reports `15.3/13.7/27.8` meV for HOMO/LUMO/Gap and the NMI paper reports 9/12 QM9 targets plus force/robustness transfer | MIT Frad and FradNMI repositories, Zenodo weights, Figshare source data, PCQM pretraining command, and RN/VRN configurations | Chemical-aware torsion/bond/angle noise produces `x_med`, coordinate Gaussian noise produces `x_fin`, and only `x_fin-x_med` is denoised; legacy Python/PyTorch/PyG and external DFT/RDKit+MMFF geometry | chemical-aware pretraining design reference; no checkpoint reuse or current torsion-state experiment |
| SliDe | PCQM4Mv2 label-free pretraining; 1,000-molecule DFT-force audit; QM9 HOMO/LUMO/Gap `13.6/12.3/26.2` meV; MD17 force transfer | MIT code, PCQM command, OpenFF parameter extraction, QM9/MD17 model links, and explicit BAT/random-slicing configs | DFT-equilibrium coordinates, OpenFF/Sage prior, GET/`N_v=128` cost, QM9 rather than PCQM Gap downstream | physics-informed pretraining and force-audit reference; future ETKDG-only adaptation only |
| CCMD paper | PCQM validation `0.0809` with global/local 3D-to-2D distillation and size-coordinated local loss; naive local-only loss degrades | no verified executable repository or checkpoint; primary paper equations/tables only | DFT teacher coordinates, prior Graphormer split, 68M-scale model, validation table versus abstract test-challenge claim | teacher-loss/control reference; no current score or implementation import |
| NVIDIA PCQM4Mv2 winner | test-challenge `0.0723`, validation `0.07145`; 39 checkpoints from 10 heterogeneous models and Huber OOF stacking | MIT repository, fold builder, Docker/per-model commands, and ensemble notebook | PCQM provided 3D SDF, RDKit image views, train+valid fold construction, 18-layer width-768 models, and 8-GPU recipes | completed delivery/ensemble reference; no current single-model score or artifact import |
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

## 6A. Direct-PCQM artifact audit: completed systems and negative controls

This section records repository-level facts that are useful for engineering
provenance, but it does not promote a codebase to a MolGap experiment. A public
checkpoint is an asset to audit; it is not permission to change the current
ETKDG, split, database, or budget contract.

### GraphGPT: graph-to-sequence implementation with released PCQM checkpoints

The [official GraphGPT repository](https://github.com/alibaba/graph-gpt)
publishes source, OGB data preparation, training and inference commands, and
four PCQM4M-v2 checkpoints through ModelScope. Its README reports no-3D,
use-3D, and latest PCQM validation claims of `0.0802`, `0.0709`, and `0.0683`,
respectively; those values remain repository/paper claims rather than a new
MolGap measurement. The implementation converts graphs to sequences using
Eulerian paths and pretrains with next-token and sequence-matching objectives.
The current README pins a modern Python 3.10/PyTorch 2.5.1/CUDA 12.4 stack,
while also documenting older tested environments.

**What is reusable.** The repository has a more complete source-to-checkpoint
lineage than many paper-only leads: preprocessing, checkpoint naming, and
supervised fine-tuning are visible. The sequence conversion is a distinct
representation hypothesis and could be a frozen-teacher or representation
probe after the active architecture question is closed.

**MolGap boundary.** The public artifact does not establish the project's
ETKDG-only train/inference geometry, and its 3D route is not a drop-in
replacement for the current GraphState input. No checkpoint, sequence cache,
or external ModelScope asset is imported.

### GraphQPT: public quantum-property pretraining package

The [GraphQPT repository](https://github.com/aidd-msca/GraphQPT) is a
reproducibility package for the atom-in-a-molecule quantum-pretraining study.
It publishes MIT-licensed code, YAML configurations, analysis notebooks, and
links to curated pretraining datasets on Zenodo. The entry point is a custom
Graphormer implementation; the README exposes a direct command of the form
`python main.py ./yaml_files/yaml_file_of_choice.yaml` rather than a hidden
training service.

**What is reusable.** The package cleanly separates atom-level quantum
pretraining, molecule-level PCQM HLG pretraining, masking pretraining, and
downstream fine-tuning. Its representation-analysis notebooks are useful
controls for checking whether a teacher changes local sensitivity or graph
spectral behavior instead of only changing one validation MAE.

**MolGap boundary.** The strongest results are TDC ADMET/HLM transfer, not a
same-contract PCQM Gap result. The atom-level source uses an external
MMFF94s/GFN2-xTB/B3LYP-def2-SVP pipeline, and the PCQM HLG pretraining is
target-family supervision rather than label-free pretraining. The code also
expects its own older Graphormer environment and curated-data paths. Grade it
**B for electronic-teacher design, C for the current screen**; no external
labels, dataset rows, or checkpoint are imported.

### OrbNet-Equi: released orbital-electronic Delta-learning package

The [OrbNet-Equi paper](https://doi.org/10.1073/pnas.2205221119), [arXiv
record](https://arxiv.org/abs/2105.14655), and [Zenodo
package](https://zenodo.org/records/6568437) expose source data, the SDC21
training dataset, neural-network code, evaluation examples, and checksums. The
implementation uses a low-cost GFN-xTB mean-field calculation to generate
atomic-orbital electronic-operator matrices, then applies an equivariant UNiTE
network. This is a real feature/teacher artifact rather than a paper-only
Delta-learning claim.

The important reusable control is explicit: direct learning is compared with
residual learning from the low-fidelity electronic estimate for QM9 frontier
properties. The paper reports a data-size crossover for LUMO and Gap under the
default features, while energy-weighted occupied/virtual density matrices keep
the Delta/direct separation. This makes the low-level feature quality, not the
word “Delta,” the falsifiable mechanism.

**MolGap boundary.** GFN-xTB operators, QM9/SDC21 theory and geometries, and
the 19 GB archive are external to the B3LYP/6-31G*/ETKDG contract. No matched
PCQM4Mv2 Gap result or graph-only path is supplied. Borrow only the
direct/scalar-residual/orbital-residual matrix and the multi-size crossover
diagnostic; do not generate GFN-xTB features or import the Zenodo package.

### Image-super-resolution electron density: public real-space teacher package

The [Nature Communications paper](https://doi.org/10.1038/s41467-025-60095-8),
[Zenodo model/code record](https://doi.org/10.5281/zenodo.15226766), and
[Figshare data release](https://figshare.com/articles/dataset/Image_Super-resolution_Inspired_Electron_Density_Prediction/25365508)
form a public model/data surface. A 3D convolutional ResNet maps a coarse
superposition of neutral atomic densities to a high-resolution density, with
positive output and electron-number normalization. The paper's QM9 density
data use PBE, GTH pseudopotentials, and GTH-TZV2P in a Gaussian--Plane-Wave /
PySCF workflow.

The model obtains energy, HOMO, LUMO, and Gap through a one-step Kohn--Sham
Fock build and diagonalization; the QM9 table reports about `11 meV` Gap MAE
for this post-diagonalization quantity. It also exposes limited-data
fine-tuning on new conformers and elements. The result is therefore a
real-space electronic-teacher and geometry-transfer artifact, not a direct
frontier-property checkpoint.

**MolGap boundary.** Uniform grids, PBE/GTH labels, QM9/water/MD roles, and
post-diagonalization evaluation differ from PCQM4Mv2 B3LYP/6-31G*/ETKDG.
Grid memory and DFT post-processing are also outside the bounded GraphState
screen. No density rows, grid files, pretrained model, or current experiment
was imported.

### 3DGrid-VQGAN: public density-grid foundation package

The [IBM repository](https://github.com/IBM/materials/tree/main/models/3dgrid_vqgan),
[Hugging Face card](https://huggingface.co/ibm-research/materials.3dgrid_vqgan),
and [ICLR paper](https://openreview.net/pdf/51c97777a512d94b366ffd7ba0d8979eba14e3c8.pdf)
expose data, pretraining, fine-tuning, inference, and embedding scripts. The
pretraining surface is a roughly `855K` neutral PubChem density-grid corpus with
`128^3` grids and a `3DGrid-VQGAN_43.pt` checkpoint. The audited data description
uses RDKit distance-geometry/force-field conformers, five MINDO3 reoptimizations,
the lowest MINDO3 structure, and RHF/STO-3G density labels.

The QM9 script has separate train/valid/test loaders, a `gap` target, one-GPU
fine-tuning settings, and an MAE output path. The paper's QM9 frontier table
contains `0.0088/0.0058/0.0057`, but the available table context does not make
their units explicit; the repository therefore cannot be used to assert an eV
result. **Use:** checkpoint/data-lineage and density-tokenization reference.
**Boundary:** external grid data, QM9 B3LYP/6-31G(2df,p), RHF/STO-3G input
density, multi-terabyte storage, and no PCQM/ETKDG result. No asset was imported.

### Graph2Mat: public sparse density-matrix and SCF-quality package

The [Graph2Mat code](https://github.com/BIG-MAP/graph2mat), [paper](https://doi.org/10.1088/2632-2153/adc871),
and [DTU MD17 record](https://data.dtu.dk/articles/dataset/MD17_data_for_graph2mat/26195285)
form a reproducible operator-learning surface. The code provides training tools,
CLI/server components, a SIESTA interface, and matrix datasets containing
Hamiltonian, overlap, density, and energy-density objects. The method maps atomic
numbers and coordinates to sparse equivariant density-matrix blocks using
MACE-style embeddings.

The repository/paper also expose the useful acceptance metrics: electron-count
error and Hamiltonian self-consistency residual, alongside SCF warm-start and
dipole/charge evaluation. These metrics are suitable for a future teacher-quality
gate. **Boundary:** SIESTA 5.0.0, pseudodojo PBE pseudopotentials, basis-dependent
matrix blocks, external coordinates, and no direct HOMO/LUMO/Gap or PCQM/ETKDG
result. No matrix data or code dependency was imported into MolGap.

### ACE density matrix: public ACE/Grassmann operator package

The [ACEsuit repository](https://github.com/ACEsuit/ACEdensitymatrix) is a Julia
package for ACE density-matrix fitting, prediction, spectral retraction, and
operator mode. Its data contract includes frame fields, density-matrix blocks,
ACE degree/order/cutoffs, fitting commands, and a link to the [DaRUS data
record](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419/DARUS-4902).
The published method uses a Grassmann retraction to map a predicted matrix back
to a physically valid projector-like manifold and reports the commutator residual
as an error/UQ signal.

**Use:** a physics-constrained operator and teacher-acceptance reference.
**Boundary:** Julia/ACE external stack, solvent/QM-MM and basis-dependent 3D
frames, omegaB97XD/6-31G(d) examples, no direct PCQM frontier result, and no
ETKDG path. No DaRUS file was downloaded.

### SMILESDFT-CLIP/SigLIP: public multimodal density-pretraining package

The [IBM implementation](https://github.com/IBM/materials/tree/main/models/smilesdft_clip)
and [model card](https://huggingface.co/ibm-research/materials.smilesdft-clip)
expose contrastive pretraining, embedding extraction, QM9 `U0` fine-tuning, and
named `SMILESDFT-CLIP_96.pt`/`SMILESDFT-SigLIP_96.pt` checkpoints. The associated
[NeurIPS workshop record](https://neurips.cc/virtual/2025/126037) reports paired
canonical SMILES and 3D density grids, random-SO(3)-rotation retrieval checks,
and functional-group retrieval diagnostics.

**Use:** frozen multimodal encoder, rotation-invariance, and train-only
contrastive-control design. **Boundary:** PubChem density grids and identities,
external density convention, no matched PCQM B3LYP/6-31G*/ETKDG frontier result,
and no current weight import. A future same-database probe would need a
train-identity filter and an unchanged random-init control.

### QMLearn: public 1-RDM gamma/delta surrogate package

The [QMLearn source](https://gitlab.com/pavanello-research-group/qmlearn),
[Nature Communications paper](https://doi.org/10.1038/s41467-023-41953-9),
[Zenodo data](https://doi.org/10.5281/zenodo.7946420), and [Zenodo code
snapshot](https://doi.org/10.5281/zenodo.8269767) expose a real electronic
surrogate implementation. The public workflow stores GTO-basis external-potential
and 1-RDM pairs, trains gamma-learning and delta-learning maps, and can evaluate
energies, forces, one-electron observables, and post-diagonalization frontier
orbitals. The code is Python/PySCF/ASE/scikit-learn oriented and includes notebooks
and training/test data for the published figures.

**Reusable:** separate gamma map, matrix-residual delta map, observable readout,
and data/engine/structure-handler boundaries. **Boundary:** molecule-specific
GTO/PySCF theory, normal-mode/Eckart geometry, small-molecule archives, no
PCQM/ETKDG path, and no current scalar Gap checkpoint. No QMLearn asset was
downloaded.

### QMLearn-SCF: public follow-up archive with force and geometry controls

The [JCTC paper](https://doi.org/10.1021/acs.jctc.5c01564), [open
preprint](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/68c9a9763e708a76498380b5/original/main.pdf),
and [Zenodo record](https://zenodo.org/records/17103131) expose reproducible
HDF5 data, HOMO/LUMO/Gap plotting scripts, learning curves, RMSE/force-correction
analysis, AIMD analysis, and cost plots. The method selection is explicit: KRR
for the external-potential-to-1-RDM map and a linear delta correction were retained
as the reliable low-cost combination. The frontier calculation still builds and
diagonalizes a Fock matrix from a predicted 1-RDM.

**Reusable:** teacher-quality gates at SCF threshold, direct versus residual
operator controls, geometry-coverage scans, and force/trajectory acceptance.
**Boundary:** about `1.3 TB` of external molecule-specific data, GTO/B3LYP
labels, non-ETKDG geometry, and no direct PCQM Gap head. No archive was downloaded.

### STRUCTURES25: public variational orbital-free DFT package

The [official repository](https://github.com/sciai-lab/structures25),
[JACS/arXiv paper](https://arxiv.org/html/2503.00443v2), and [documentation](https://sciai-lab.github.io/structures25/)
provide a current code/model surface with inference-only installation, QM9 and
QMugs model identifiers, data-generation tools, density-optimization workflows,
and replication instructions. The `OFData` schema explicitly stores positions,
atomic numbers, density coefficients, basis metadata, electron count, energy
labels, gradients, and SCF-iteration fields. The model learns a variational
energy functional and obtains density gradients through autodiff; perturbed
effective-potential samples are a first-class data-generation option.

**Reusable:** density/energy auxiliary objective, perturbed-state coverage,
electron-count normalization, and stable variational teacher acceptance.
**Boundary:** explicit 3D coordinates and basis coefficients, PBE/QMugs roles,
no direct HOMO/LUMO/Gap head, and external LGPL-3.0 software/model contract.
No code, model, or data was imported.

### KineticNet: paper-complete derivative teacher, no public data archive

The [JCP paper](https://doi.org/10.1063/5.0158275) and [arXiv
HTML](https://arxiv.org/html/2305.13316) provide a detailed E(3)-equivariant
grid model and a reproducible training specification: BLYP/cc-pVDZ, random
external-potential matrix perturbations, separate kinetic-energy and derivative
fields, and small-molecule test metrics. The paper's data-availability statement
offers the data from the corresponding author on reasonable request. No official
repository, checkpoint, or public archive was verified, so KineticNet is not a
public-code candidate.

**Reusable:** derivative-loss design, off-ground-state coverage, atomic-cusp
handling, and the requirement to evaluate iterative stability rather than only
field MAE. **Boundary:** real-space grids, external labels, author-request data,
and no HOMO/LUMO/Gap head. No code or data was imported.

### M-OFDFT: public residual/gradient-label and pretrain--fine-tune surface

The peer-reviewed [Nature Computational Science paper](https://doi.org/10.1038/s43588-024-00605-8),
[arXiv PDF](https://arxiv.org/pdf/2309.16578), [Zenodo implementation](https://doi.org/10.5281/zenodo.10616893),
and [Figshare model/data collection](https://doi.org/10.6084/m9.figshare.c.6877432)
provide a full implementation surface. The paper's code/data route exposes
atom-centred density coefficients, an APBE-residual KEDF, projected KS-SCF
gradient labels, local-frame and normalization transforms, density optimization,
and explicit pretrain/from-scratch/fine-tune comparisons. Nature's availability
statement separately confirms the implementation DOI and trained checkpoints.

**Reusable:** residual-target packaging, teacher-gradient quality gates, and the
pretraining control matrix. **Boundary:** PBE/6-31G(2df,p), QM9/QMugs/MD17/
chignolin rows, explicit density coefficients/3D geometry, no direct frontier
head, and external model/data terms. No external artifact was downloaded.

### QH9, nablaDFT, and ∇²DFT: public operator-data surfaces

The [QH9 paper](https://arxiv.org/html/2306.09549), official
[AIRS/QHBench code](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9),
and [Zenodo record](https://zenodo.org/records/8274793) expose loaders,
Hamiltonian/overlap objects, stable and dynamic splits, baselines, and pretrained
model loading. [nablaDFT](https://github.com/AIRI-Institute/nablaDFT) adds a
public Hamiltonian/energy database and model registry. [∇²DFT](https://arxiv.org/html/2406.14347)
adds a large public density/Hamiltonian/wavefunction dataset with metadata for
frontier quantities.

**Reusable:** method-aware operator schemas, independent acceptance metrics,
SCF-initialization checks, chunked data manifests, and a separation between
operator labels and post-diagonalization observables. **Boundary:** the public
records use explicit 3D geometries and B3LYP/def2-SVP or ωB97X-D/def2-SVP-type
contracts; no QH9/∇²DFT row or checkpoint is imported into PCQM/ETKDG.

### DEQHNet, QHFlow, QHFlow2, and QHNetV2: Hamiltonian model families

[DEQHNet](https://github.com/Zun-Wang/DEQHNet) supplies a fixed-point Hamiltonian
network with public code. [QHFlow](https://github.com/seongsukim-ml/QHFlow) and
[QHFlow2](https://github.com/seongsukim-ml/QHFlow2) expose flow-matching Hamiltonian
models; QHFlow2 additionally publishes [QH9](https://huggingface.co/ksusu/QHFlow2-QH9)
and rMD17 checkpoint cards and reports HOMO/LUMO/Gap after diagonalization.
[QHNetV2](https://arxiv.org/html/2506.09398) is available through
[AIRS](https://github.com/divelab/AIRS) and documents a lower-cost SO(2)-frame
route.

**Reusable:** fixed-point iteration, H-to-eigenvalue readout, explicit stable/
dynamic split evaluation, and separation of matrix error from orbital error.
**Boundary:** these are external Hamiltonian/overlap targets with explicit 3D
inputs; QHFlow2's public checkpoint is a teacher/probe asset, not a current
random-init or warm-start import.

### NeuralSCF and HamEvo: public self-consistent and operator-Delta code

[NeuralSCF](https://github.com/songfeitong/neuralscf) and its
[Science Data Bank release](https://doi.org/10.57760/sciencedb.37834) expose
SCF-trajectory pretraining, implicit fixed-point fine-tuning, Pulay iteration,
and density-coefficient data. [HamEvo](https://github.com/axdfhj/HamEvo_official)
and its [Hugging Face data](https://huggingface.co/datasets/ZJUSCL/hamevo-data)
expose staged SCF-transition pretraining, local `Delta H` updates, Anderson
forward iteration, Broyden backward/implicit differentiation, and size/
functional-transfer experiments.

**Reusable:** a residual must be evaluated on the fixed-point trajectory; a
teacher should be accepted by frontier error, density/Hamiltonian residual, and
iteration stability rather than matrix MAE alone. **Boundary:** the releases
use external basis/functional/coordinate contracts and do not provide a direct
PCQM4Mv2 ETKDG Gap head. No code, data, or checkpoint was downloaded.

### QHFlow2 upstream efficiency and loss references

[WANet/WALoss](https://arxiv.org/pdf/2502.19227) reports an overlap-aware loss
in the ground-truth eigenbasis and [SPHNet](https://proceedings.mlr.press/v267/luo25l.html)
publishes sparse pair/tensor-product gates and adaptive sparsity scheduling with
an archived [MIT package](https://github.com/microsoft/SPHNet). These are useful
for operator-quality acceptance and memory/compute design. They do not establish
a current scalar PCQM Gap improvement and are not admitted as a new screen.

### HELM: pretraining evidence without a public artifact

[HELM](https://arxiv.org/html/2510.00224) is retained because it explicitly
compares Hamiltonian pretraining with frozen and fine-tuned energy heads on
OMol_CSH_58k. The paper currently marks code and data as forthcoming, and its
energy/theory/geometry contract is external. It is a protocol reference, not a
downloadable checkpoint or a current initialization.

### PySCFAD and the Atomistic Cookbook: differentiable ML/QM integration

The [JCTC paper](https://doi.org/10.1021/acs.jctc.5c00522), open
[preprint](https://arxiv.org/abs/2504.01187),
[Atomistic Cookbook example](https://atomistic-cookbook.org/examples/hamiltonian-qm7/hamiltonian-qm7.html),
and [PySCFAD package](https://github.com/fishjojo/pyscfad) provide a completed
software path for differentiating through a predicted effective Hamiltonian.
The example includes direct Hamiltonian fitting, indirect property losses,
multi-target constraints, and a minimal-basis-to-large-basis transfer setup.

**Reusable:** a clean separation between model output, electronic-structure
post-processing, and observable-specific acceptance; this is particularly
useful for testing HOMO/LUMO/Gap consistency after diagonalization. **Boundary:**
QM7/QM9, STO-3G/def2-TZVP, explicit basis integrals, and a JAX/PySCFAD stack
are external. No package, model, or row is imported into MolGap.

### HamGNN: public equivariant Hamiltonian source/data/checkpoint surface

The [npj Computational Materials paper](https://doi.org/10.1038/s41524-023-01130-4),
[official GitHub repository](https://github.com/QuantumLab-ZY/HamGNN),
[Zenodo models](https://doi.org/10.5281/zenodo.8147631), and
[Zenodo training data](https://doi.org/10.5281/zenodo.8157128) form a completed
operator artifact. The package exposes Hamiltonian reconstruction, prediction,
and downstream eigenvalue/band workflows; the paper covers molecular and solid
transfer rather than a scalar PCQM Gap benchmark.

**Reusable:** equivariant block schemas, model/data/checkpoint manifests, and
operator-to-spectrum checks. **Boundary:** tight-binding/ab-initio Hamiltonian
matrices, explicit 3D/material basis roles, GPL-3.0 code/data surface, and no
matched B3LYP/6-31G*/ETKDG frontier result. Nothing was downloaded.

### HAMSTER: public physics-model-to-DFT Delta package

The [Nature Communications paper](https://doi.org/10.1038/s41467-026-70865-7),
[MIT Hamster.jl code](https://github.com/TheoFEM-TUM/Hamster.jl), and
[188-GB Zenodo data release](https://doi.org/10.5281/zenodo.18485403) expose a
complete physics-informed Delta-learning workflow. The model corrects a
tight-binding Hamiltonian with an ML residual, separates TB/ML/SOC blocks, and
stores input/output/checksum artifacts for GaAs, CsPbBr3, and MAPbBr3.

**Reusable:** explicit low-fidelity model, residual target, versioned data/code,
and physical eigenvalue/band-gap acceptance. **Boundary:** periodic inorganic
materials, PBE/VASP/SOC/TB labels, Julia implementation, and 188-GB external
data. It is an artifact/Delta reference only; no code, data, or weights were
imported.

### Tetrahedral Molecular Pretraining: public same-database 3D pretraining asset

The [Pattern Recognition paper](https://doi.org/10.1016/j.patcog.2025.112638)
and [MIT repository](https://github.com/sunyuancheng/Tetrahedral-Molecular-Pretraining)
publish a completed tetrahedral pretraining route. The repository exposes the
LEGO pretraining/fine-tuning path and a checkpoint; the paper defines
non-overlapping one-hop tetrahedral units, global orientation prediction, and
local structure reconstruction on PCQM4Mv2 3D structures. It reports a
`0.0817` versus `0.0910` PCQM downstream comparison while excluding Gap labels
from the pretraining objective.

**Reusable:** teacher/pretraining role separation, unit-level corruption, and
checkpoint/configuration manifests. **Boundary:** DFT-simulated coordinates,
large Transformer-M scale, legacy dependencies, and unresolved downstream
split/validation lineage. The released artifact is not an ETKDG train/inference
model and no checkpoint was downloaded.

### Meyer derivative training: open paper, no production package

The [JCTC article](https://doi.org/10.1021/acs.jctc.0c00580) and [PMC full
text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7482319/) give a complete 1D
method description and supporting tables, but no molecular production code or
checkpoint. The result is kept as a negative-control reference: joint energy and
functional-derivative training improves local fits, yet noisy derivatives still
break unconstrained optimization without projection or a physical penalty.

**Reusable:** failure-aware auxiliary-loss acceptance. **Boundary:** synthetic
1D fermions, no molecule/geometry/frontier target, and no current experiment.
No code or data was imported.

### DGT: dual atom/bond graph pretraining asset

The [DGT Nature Communications paper](https://www.nature.com/articles/s41467-026-75005-9)
publishes a dual atom/bond Transformer with cross-level fusion, optional 3D
features, and relative/spatial encodings. The [MIT repository](https://github.com/zhangsy-ryan/DGT),
[Zenodo code archive](https://doi.org/10.5281/zenodo.20009509), and [Figshare
source-data record](https://doi.org/10.6084/m9.figshare.30665129) provide a
substantive public artifact surface.

The PCQM role is pretraining: the paper uses random `10K`, `100K`, and `1M`
PCQM4Mv2 subsets to predict Gap, then transfers the weights to QM9. The
reported `10K` downstream QM9 HOMO/LUMO MAEs are `0.0306/0.0240 eV`, not a
PCQM Gap leaderboard number, and the paper reports no significant gain from
larger pretraining subsets. The visible repository configurations are QM9
configurations; no PCQM pretraining config or checkpoint is exposed in the
visible tree.

**MolGap boundary.** QM9 uses B3LYP/6-31G(2df,p), and the 3D comparison uses
DFT/MMFF/UFF geometry rather than ETKDG. Dense pair encodings are quadratic,
and the bond-state path overlaps the existing EdgeState/GraphState work. Keep
DGT as a complete pretraining and dual-graph reference, but do not import its
weights, geometry, or database rows and do not treat its QM9 result as direct
PCQM evidence.

### QuantumCanvas: two-body electronic fields with a reproducible artifact surface

The [QuantumCanvas paper](https://arxiv.org/abs/2512.01519), [MIT repository](https://github.com/KurbanIntelligenceLab/QuantumCanvas), and [Zenodo release](https://doi.org/10.5281/zenodo.20631934) provide a more complete public asset than an abstract-only two-body proposal. The repository exposes a `dataset_combined.npz` bundle, an MD5, loaders, and `REPRODUCE.md`; the paper defines `2,850` element pairs over `75` elements and tests eight graph/vision/fusion models across element-pair-disjoint splits and three seeds.

The useful engineering lesson is the artifact contract: fixed-size electronic
fields, an explicit dataset bundle, a checksum, loader code, and a composition-
held-out evaluation. The reported GATv2 gap MAE is `0.201 +/- 0.020 eV`, and
the paper transfers pretraining to QM9, MD17, and CrysMTM. The image branch is
not a current MolGap candidate because the actual labels are SCC-DFTB/DFTB+
PTBP, the samples are isolated diatomics with explicit coordinates, and the
paper/README target-schema counts disagree (`18/22` versus `37` keys). Keep it
as a two-body electronic-teacher and manifest reference only.

### TMC-Delta-ML: source-complete residual-learning code

The [TMC-Delta-ML repository](https://github.com/uiocompcat/TMC-Delta-ML) is
MIT-licensed and links to the peer-reviewed [paper](https://doi.org/10.1002/chem.71487),
the [tmQMg graph dataset](https://github.com/uiocompcat/tmQMg), and a
[Zenodo low-fidelity graph archive](https://doi.org/10.5281/zenodo.18348669).
Unlike a conceptual Delta-ML description, the README exposes `bench`, `lsda`,
and `pbe0` switches, fixed CSD-ID train/validation/test split files, automatic
tmQMg retrieval, and a target-specific HOMO--LUMO path. It also distinguishes
properties that use atomic baseline fitting from Gap, polarizability, and
dipole, which do not.

The scientific contract is transition-metal tmQMg: GFN2-xTB geometry, u-NatQG
graphs, and low-fidelity LSDA/LANL2DZ or PBE0-D3BJ/def2-TZVP graph inputs for
PBE0-D3BJ/def2-TZVP or related high-fidelity targets. The paper reports better
accuracy/data efficiency and cheaper-fidelity tradeoffs than its benchmark,
but it does not establish PCQM4Mv2 B3LYP/6-31G* or ETKDG performance. This is a
high-grade Delta-learning and cost-accounting reference, not a current MolGap
weight/data import.

### SelectedML: executable chemical-family stratification and Delta control

The [Selected Machine Learning paper](https://doi.org/10.1039/D2MA00742H),
[arXiv record](https://arxiv.org/abs/2110.02596), and [public repository](https://github.com/b3rn4rdm/SelectedML)
form a coherent older but directly relevant artifact. The code performs
structure-based frequency analysis, writes class-labelled CM/BoB/SLATM data,
and exposes cross-validation/learning-curve scripts for QM7 and QM9. It also
contains a QM7b direct/Delta-QML path with explicit low- and high-level property
arguments and three-seed controls.

The paper's class rule is chemically interpretable: aromatic-ring/carbonyl,
singly unsaturated, and saturated molecules are modelled separately. The
reported learning-curve improvement is not a PCQM claim, but it is useful
evidence that a global Gap regressor can hide distinct error regimes. For
MolGap, the safe reuse is a post-hoc family-conditioned MAE/bias/residual audit
on the same official rows. A class-conditioned head or mixture of experts would
need a new paired protocol, a train-only structure rule, and a global control;
the QM7/QM9 KRR code and labels cannot be imported as a pretrained model.

### HLP-Stack: public descriptor ensemble, not a valid current baseline

The [RSC Advances paper](https://doi.org/10.1039/D5RA08007J), [PMC full record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959570/), and [HLP-STACK repository](https://github.com/college-of-pharmacy-gachon-university/HLP_STACK) expose the code, raw/processed data, notebooks, saved models, figures, and environment. The paper combines RDKit 2D descriptors with QM9 3D quantum descriptors, selects `51` features from an initial `221`, and stacks Random Forest, XGBoost, Extra Trees, and Gradient Boosting with a linear meta-learner.

The reported HOMO/LUMO test errors are extremely small (`3.219e-4` and
`1.903e-4 Eh`, (R^2 \approx 0.9999)). This is precisely why the artifact is
useful as an audit: the paper states that the 3D descriptor source is
B3LYP/6-31G(2df,p) QM9, and the descriptor list includes quantum/electronic
quantities that are not computable from the present input without an extra QC
calculation. The result is neither PCQM4Mv2 nor an ETKDG-only inference path,
and the near-perfect train/validation summaries must not be treated as a
portable baseline without feature-level leakage and split verification.

Keep HLP-Stack for the negative question “which features would require target-
adjacent quantum information?” A legal future descriptor control would need
to restrict every feature to the same graph plus ETKDG coordinates and compare
against a descriptor-free GraphState control. Do not import its models or
QM9 quantum features.

### PCQM 3D-prior distillation: reproducibility package with negative smoke evidence

The [PCQM 3D-prior distillation repository](https://github.com/A-SHOJAEI/pcqm4mv2-3d-prior-distillation)
is a small but explicit implementation of a 3D EGNN teacher, a 2D Graph
Transformer student, and separate supervised, prediction-level KD, and
feature-level KD losses. The configuration records official OGB split keys and
offers `dft` or RDKit `ETKDGv3+UFF` teacher geometry. It also writes split/data
manifests, runtime metadata, per-run checkpoints, predictions, and paired
bootstrap metrics.

The evidence is negative but valuable: the checked-in `smoke` artifact uses
synthetic data, one seed, and a two-epoch teacher. It reports validation MAE
`0.882684` for the GIN virtual-node baseline, `1.036029` for the distilled
student, and `1.035819` for the no-distillation student; the repository itself
states that this is not a PCQM4Mv2 leaderboard run. Distillation therefore did
not improve the smoke baseline, and the two student variants are effectively
indistinguishable at this scale.

**What is reusable.** Keep teacher/student/baseline switches, no-distillation
controls, geometry-source fields, split manifests, per-run artifacts, and
paired bootstrap intervals. **What is not established:** a positive PCQM Gap
gain, a full-data result, or permission to use the DFT geometry mode. The
repository is **B for implementation structure and C for scientific
performance**; its synthetic failure is a reason to require a real teacher
quality gate before any future ETKDG distillation budget.

### TGT: completed learned-distance/teacher pipeline

The [official TGT repository](https://github.com/shamim-hussain/tgt) exposes
PCQM preprocessing, an inference notebook, and [TGT-At/TGT-Agx2 weights](https://huggingface.co/shamim-hussain/tgt)
plus a [preprocessed data release](https://huggingface.co/datasets/shamim-hussain/pcqm).
The documented pipeline has three stages: distance prediction, Gap pretraining
with noisy coordinates, and fine-tuning using predicted distances. The README
reports TGT-At plus RDKit coordinates at `0.0671` validation and `0.0683`
test-dev, TGT-At without RDKit coordinates at `0.0686` validation and `0.0698`
test-dev, and TGT-Agx2 plus RDKit coordinates at `0.0682` validation.

**What is reusable.** This is a completed systems reference for separating
geometry prediction from property training, and for storing data, weights, and
inference entry points as separate artifacts. It also gives a concrete upper
bound on what a geometry-supervised pipeline can achieve under a different
contract.

**MolGap boundary.** The README calls the auxiliary input “RDKit Coords” and
does not prove the exact project ETKDG method or current atom-ordering policy.
The 102M-parameter stages and three-stage learned-distance workflow are not a
bounded single-model screen. The weights and PCQM-derived data remain external
references, not current inputs.

### Graphcore GPS++: durable IPU-specific implementation

The [Graphcore PCQM4Mv2 repository](https://github.com/graphcore/ogb-lsc-pcqm4mv2)
publishes custom grouped gather-scatter/static operations, PCQM preparation,
training/inference scripts, notebooks, and released 11M/22M/44M configurations.
Its README reports approximately `0.090`, `0.082`, and `0.077` validation MAE
for those scales, and describes a challenge ensemble of seven models. The code
and checkpoints are useful as a completed artifact layout; the repository also
states the relevant checkpoint licensing separately from the MIT code.

**What is reusable.** Sharded preprocessing, model-zoo metadata, checkpoint
boundaries, and the separation of training from inference are directly useful
for future durable artifacts. The numbers are not compared against the current
GraphState screen because the systems, split reporting, and accelerator are
different.

**MolGap boundary.** The implementation relies on Graphcore IPU/Poplar custom
operations and is not a portable PyTorch candidate for the bounded A100
screen. No IPU checkpoint or PCQM feature cache is imported.

### MooseML: deployment reference, not benchmark evidence

The [MooseML application](https://github.com/MooseML/homo-lumo-gap-predictor)
is a useful completed packaging example: Streamlit/Docker deployment, batch
CSV inference, SQLite logging, and a live Hugging Face Space. Its model is a
GINE-style graph encoder with global mean pooling, a dense head, and six RDKit
physicochemical descriptors; the README says the training set is OGB
PCQM4Mv2 and that Optuna was used for tuning.

The repository exposes no peer-reviewed paper, released checkpoint, or
independently checkable PCQM metric. It is therefore **C** evidence: borrow
deployment/monitoring ideas only, and do not treat the app as a scientific
baseline or import its descriptors into the current model.

### Spectral-temporal curriculum: negative public claim

The [spectral-temporal-curriculum repository](https://github.com/A-SHOJAEI/spectral-temporal-curriculum-molecular-gaps)
claims spectral graph wavelets, curriculum training, PCQM4Mv2 training, and a
test MAE of `0.268 eV` with `73,545` predictions. The visible repository surface
has zero stars, three commits, no linked primary paper or independently
traceable run, and no visible released checkpoint. The score is far below the
audited PCQM references, but the more important issue is missing provenance:
there is no basis for deciding split, target role, coordinate source, or metric
reproduction.

**Disposition: C / negative artifact.** Keep it as a warning that a complete-
looking repository and a number are not sufficient evidence. Do not spend
compute, download a checkpoint, or use its result to reprioritize the current
screen.

### PET-MAD-DOS: electronic teacher with data, scripts, and checkpoint surfaces

The [PET-MAD-DOS paper](https://doi.org/10.1039/D5DD00557D),
[Materials Cloud reproduction record](https://doi.org/10.24435/materialscloud:gs-z7),
[UPET repository](https://github.com/lab-cosmo/upet), and
[Hugging Face model](https://huggingface.co/lab-cosmo/pet-mad-dos) expose a
complete electronic-structure artifact. The release contains `.xyz` data with
`number of electrons`, `gap`, `DOS`, and reliability `mask`, plus training,
fine-tuning, UQ, LoRA, DOS-denoising, and gap-extraction scripts. The model is
a PET Transformer trained on MAD and predicts a fixed-grid DOS from atomic
structures; a CNN or a physical DOS procedure turns the output into a gap.

This is a strong packaging reference because the electronic object, energy
grid, reliability mask, electron count, model material, and reproduction
scripts are released together. It is not a current MolGap model asset: the
release uses heterogeneous MAD structures and PBEsol/Quantum Espresso rather
than B3LYP/6-31G* PCQM labels, and its DOS-derived gap is not a released
HOMO/LUMO pair under the ETKDG contract. No model or data was downloaded.

### FieldMACE: public long-range message implementation

The [FieldMACE repository](https://github.com/rhyan10/FieldMACE) contains the
extended MACE source tree, publication checkpoints/logs, training scripts,
validation indices, and an explicit `--foundation_model` /
`--multipole_max_ell` interface. The associated [Figshare archive](https://figshare.com/articles/dataset/Models_data_and_code_for_publication_Incorporating_Long-Range_Interactions_via_the_Multipole_Expansion_into_Ground_and_Excited-State_Molecular_Simulations_/28497857)
holds the publication data and models. Its implementation is specifically
for QM/MM: multipole features derived from the MM environment enter the
equivariant message update, and energy/force training is used for ground- and
excited-state simulations.

The repository is therefore useful for source-level study of higher-order
long-range feature injection and foundation-model transfer. It is not a
drop-in HOMO/LUMO/Gap baseline: explicit MM point charges, external solvated
geometries, and energy/force targets violate the current PCQM/ETKDG role map.

### MACE-POLAR-1: released charge/spin-aware foundation checkpoint

The [official MACE foundation release](https://github.com/ACEsuit/mace-foundations/releases)
publishes MACE-POLAR-1 model artifacts, and the [paper](https://arxiv.org/html/2602.19411)
specifies the architecture and OMol25 training contract. The model combines a
local MACE backbone with non-self-consistent polarizable field updates,
Gaussian-smeared atomic multipoles, and global Fukui equilibration for total
charge and spin. The public release identifies medium/large receptive fields
and ASL distribution terms.

The code/checkpoint surface is complete enough for a future frozen-teacher
audit, but the public evidence is for hybrid-DFT energy/force foundation
training and physical charge/spin response—not a PCQM4Mv2 frontier-orbital
head. Its weights and OMol25 rows remain external; an ETKDG inference smoke
would not, by itself, establish transfer benefit.

### MACE-H: reproducible operator-level electronic model

The [MACE-H repository](https://github.com/maurergroup/MACE-H) is MIT licensed
and exposes preprocessing, training/evaluation, inference/post-processing,
environment files, and a [Zenodo configuration/container package](https://doi.org/10.5281/zenodo.15223696).
The [paper](https://arxiv.org/html/2508.15108) predicts local KS Hamiltonian
blocks with high-body-order messages and node-degree expansion; Julia tools
diagonalize predicted reciprocal-space matrices for bands and DOS.

This is a finished electronic-operator workflow, but it is for periodic 2D
materials and bulk Au with OpenMX/FHI-aims basis conventions, not isolated
PCQM molecules. It is a high-body-order/operator-supervision reference only;
its matrix or band errors must not be reported as molecular Gap evidence.

### CheMeleon: public descriptor-pretraining data, weights, and loader

The [CheMeleon repository](https://github.com/JacksonBurns/chemeleon) links the
[paper](https://arxiv.org/html/2506.15792), a [Zenodo training-data
release](https://doi.org/10.5281/zenodo.15733574), a [Zenodo model-weight
release](https://doi.org/10.5281/zenodo.15426600), and the [Chemprop transfer
documentation](https://chemprop.readthedocs.io/en/main/chemeleon_foundation_finetuning.html).
The public route is unusually concrete: a one-million-molecule PubChem
descriptor-pretraining surface, a released D-MPNN foundation checkpoint, and a
documented `--from-foundation CheMeleon` downstream loading path. The paper
describes dynamic 85% Mordred-descriptor masking, retention of the pretrained
D-MPNN, removal of the descriptor head, and end-to-end fine-tuning with a fresh
task head.

The repository also exposes an important feasibility warning: the full
descriptor/pretraining reproduction is estimated at roughly 500 CPU-hours plus
1000 GPU-hours and more than 1 TB of checkpoint/storage footprint. That is a
foundation-building cost, not the cost of a small downstream fine-tune. The
artifact is therefore a strong implementation reference but not a reason to
download external weights into MolGap. It has no verified matched PCQM4Mv2
B3LYP/6-31G* Gap result, uses external PubChem molecules and graph/SMILES
inputs, and does not establish ETKDG train/inference consistency.

**Safe use:** borrow the head-discard/encoder-transfer layout, descriptor
manifest, missing-value audit, and paired random-init control design. If a
future experiment is authorized while keeping the database fixed, descriptors
must be generated from the PCQM training split only and the descriptor list
must be frozen before any validation/test access. No external CheMeleon data,
weights, or current run was used.

### Zatom-1: complete 3D foundation code and checkpoint surface

The [Zatom-1 repository](https://github.com/Zatom-AI/zatom) links the
[paper](https://arxiv.org/html/2602.22251) and a [Zenodo checkpoint
record](https://zenodo.org/records/19766997) containing multiple generative,
property-prediction, QM9-only, joint, and non-pretrained checkpoints. The
repository provides dataset download, training, evaluation, and explicit QM9
commands whose property list includes `homo`, `lumo`, and `gap`. This is a
complete artifact surface rather than a paper-only foundation-model claim.

The implementation is a large 3D flow Transformer: it operates on atom types
and explicit coordinates, with material lattice modalities when applicable.
The paper's controls show why a checkpoint must not be treated as a generic
warm start: whole-trunk unfreezing harms generative validity, whereas frozen
trunk and LoRA are distinct transfer contracts. The full project reports about
20,000 GPU-hours and 1 TB of local storage, and the main 80M/160M/300M variants
are far outside the current bounded screen.

Zatom is therefore valuable for checkpoint manifests, multi-head transfer
controls, and negative-transfer analysis, but not as a MolGap model asset. Its
frontier evidence is QM9/Matbench, its energy/force evidence is OMol25/MPtrj,
and no matched PCQM4Mv2 B3LYP/6-31G* Gap or ETKDG train/inference route is
published. No Zatom data or weight was downloaded.

### Artifact-level comparison

| Artifact | What is actually public | Evidence grade | Safe MolGap use |
|---|---|---:|---|
| GraphGPT | Source, preprocessing, training/inference commands, four PCQM checkpoints | B | Sequence/checkpoint provenance reference; no current initialization |
| GraphQPT | MIT code, YAMLs, analysis notebooks, and Zenodo curated data links | B/C | Electronic-teacher design and representation-analysis reference; no current labels or weights |
| OrbNet-Equi | Zenodo source/data/code archive, SDC21 data, evaluation examples, and checksums | A/B | AO-electronic Delta/reference matrix; GFN-xTB/QM9/SDC21 and non-ETKDG contract prevent current import |
| Image-super-resolution density | Zenodo model/code record, CC-BY Figshare data, density ResNet, and QM9 one-step property route | A/B | Real-space density-teacher and geometry-transfer reference; PBE/GTH grids and post-diagonalization contract prevent current import |
| 3DGrid-VQGAN | IBM source, pretraining/fine-tuning/inference scripts, Hugging Face checkpoint, QM9 target loader, and density-grid description | A/B | Density-grid foundation and frozen-teacher reference; RHF/STO-3G grids, QM9 theory, unit-ambiguous frontier table, and multi-terabyte corpus prevent current import |
| Graph2Mat | MIT code, CLI/server/SIESTA interface, sparse matrix data surface, and DTU MD17 archive | A/B | Sparse electronic-operator and self-consistency/UQ reference; SIESTA basis/pseudopotential and no frontier head prevent current import |
| ACE density matrix | Julia ACE package, fitting/prediction/retraction code, frame schema, and DaRUS learning/prediction archives | A/B | Grassmann/projector-constrained operator reference; solvent/QM-MM/basis/3D contract prevents current import |
| SMILESDFT-CLIP/SigLIP | IBM pretraining/embedding/QM9 scripts, named CLIP/SigLIP checkpoints, and model card | A/B | Multimodal SMILES--density and rotation-invariance reference; PubChem density pairs and no PCQM frontier result prevent current import |
| QMLearn | GitLab source, Nature paper, Zenodo GTO/1-RDM data, code snapshot, notebooks, and post-Fock property workflow | A/B | Gamma/δ operator-teacher and observable-readout reference; molecule-specific GTO/PySCF/Eckart contract prevents current import |
| QMLearn-SCF | Peer-reviewed follow-up, open preprint, Zenodo HDF5 archive, frontier plotting, learning-curve, force and AIMD scripts | A/B | SCF-threshold, residual, force/coverage and diagonalization-aware teacher reference; 1.3-TB GTO archive prevents current import |
| STRUCTURES25 | Official source, QM9/QMugs models, `OFData` schema, data-generation and density-optimization workflows, replication guide | A/B | Variational density/energy teacher and perturbed-state coverage reference; PBE/QMugs/basis/explicit-3D contract prevents current import |
| KineticNet | JCP/arXiv paper and detailed architecture/training specification; no verified public source, checkpoint, or data archive | A/B | Derivative-aware grid teacher and off-ground-state coverage reference; BLYP/cc-pVDZ, author-request data, no frontier head |
| M-OFDFT | Nature paper, Zenodo implementation, Figshare model/data collection, density-optimization and pretrain/fine-tune surface | A/B | Residual KEDF, projected-SCF gradients and explicit pretraining controls; PBE/QM9/QMugs/MD17/3D contract prevents current import |
| QH9 / AIRS-QHBench | QH9 paper, AIRS loaders/baselines, Zenodo data, stable/dynamic splits, and pretrained-model surface | A/B | Public operator benchmark and acceptance reference; B3LYP/def2-SVP explicit-3D contract prevents current import |
| nablaDFT / ∇²DFT | Public Hamiltonian/energy or density/Hamiltonian/wavefunction databases and loaders | A/B | Large operator-data schema and post-diagonalization metric reference; external theory/geometry prevents current import |
| DEQHNet / QHFlow / QHFlow2 | Public fixed-point/flow Hamiltonian code; QHFlow2 QH9/rMD17 Hugging Face checkpoints | A/B | Frontier-aware Hamiltonian teacher/probe and stable/dynamic evaluation; no current ETKDG or scalar PCQM contract |
| NeuralSCF / HamEvo | Public SCF trajectory/density or `Delta H` fixed-point code/data | A/B | Self-consistency and operator-Delta design; external basis/functionals/coordinates prevent current import |
| QHNetV2 / SPHNet / WANet | AIRS QHNetV2 implementation, archived SPHNet package, overlap-aware WALoss paper | A/B | SO(2), sparse-gate, and operator-loss efficiency references; no current frontier result under ETKDG |
| HELM | Paper-level Hamiltonian pretraining and frozen/fine-tuned head comparison | B | Pretraining protocol reference only; paper marks code/data TBA |
| PySCFAD / Atomistic Cookbook | Apache-2.0 differentiable PySCFAD package and runnable Hamiltonian indirect-target tutorial | A/B | Differentiable ML/QM and multi-target readout reference; QM7/QM9, basis, and integral contract prevents current import |
| HamGNN | Official E(3) code, Zenodo models/data, and Hamiltonian reconstruction/band tools | A/B | Completed operator artifact; materials/tight-binding basis, explicit 3D, and no current scalar PCQM result prevent current import |
| HAMSTER | MIT Julia package, Zenodo source data, and TB/ML/SOC Delta workflow | A/B | Physics-informed Hamiltonian Delta packaging; periodic PBE/VASP/SOC/materials contract prevents current import |
| Tetrahedral Molecular Pretraining | MIT code/checkpoint, PCQM 3D tetrahedral pretraining commands, and downstream fine-tuning path | A/B | Same-database pretraining reference; DFT-coordinate/legacy-stack/validation-lineage and non-ETKDG contract prevent current import |
| Meyer derivative training | Open JCTC/PMC paper, supporting equations/tables; no production code/checkpoint verified | C | 1D derivative-supervision negative control; no molecular target or current import |
| DGT | MIT code, Zenodo code archive, Figshare source data, and QM9 configuration surface; PCQM Gap pretraining protocol | B | Complete pretraining/dual-graph reference; no direct PCQM Gap score or current initialization |
| QuantumCanvas | MIT code, CC-BY Zenodo NPZ, MD5, loaders, and `REPRODUCE.md` | B | Two-body electronic-teacher and data-manifest reference; SCC-DFTB/DFTB+ and diatomic roles prevent current import |
| TMC-Delta-ML | MIT code, Zenodo low-fidelity graph archive, tmQMg auto-download, and fixed CSD-ID splits | A/B | Completed Delta-learning/cost-control reference; transition-metal, u-NatQG, and non-ETKDG contract prevent current import |
| SelectedML | Public functional-group frequency analysis, CM/BoB/SLATM data preparation, CV/learning-curve scripts, and direct/Delta-QML paths | A/B | Chemical-family stratification and direct-versus-Delta control reference; QM7/QM9 KRR roles prevent current import |
| HLP-Stack | Raw/processed QM9 data, notebooks, saved models, figures, and Conda environment | B/C | Public descriptor artifact but DFT-derived 3D features and near-perfect metrics require a leakage/provenance audit; no current import |
| PET-MAD-DOS | Materials Cloud data/scripts, `.xyz` DOS/mask schema, UPET API, and public Hugging Face model | A/B | Complete DOS/band-gap teacher packaging and UQ/LoRA reference; PBEsol/MAD/periodic roles prevent current import |
| FieldMACE | Extended MACE source, publication checkpoints/logs, validation indices, Figshare data/models | A/B | Long-range multipole message and foundation-transfer implementation reference; QM/MM energy/force contract prevents current import |
| MACE-POLAR-1 | Official foundation release with M/L checkpoints/configuration and MACE loader path | A/B | Charge/spin-constrained electrostatic teacher reference; OMol25 hybrid-DFT/3D energy-force contract prevents current import |
| MACE-H | MIT source, preprocessing/training/evaluation/inference tools, Zenodo configs/containers | A/B | Operator-level KS Hamiltonian and high-body-order reference; periodic materials/basis contract prevents current import |
| CheMeleon | Official repository, one-million-molecule training-data release, released foundation weights, and Chemprop fine-tuning path | A/B | Descriptor-pretrained D-MPNN transfer reference; PubChem/Mordred/SMILES contract, no matched PCQM Gap result, and high foundation cost prevent current import |
| Zatom-1 | Official 3D flow code, dataset loaders, evaluation commands, and Zenodo generative/property checkpoints | A/B | Frozen-trunk/LoRA and multi-head 3D foundation reference; QM9/Matbench/OMol25/MPtrj and explicit-coordinate contract prevent current import |
| PCQM 3D-prior distillation | Official split/geometry manifests, teacher/student/KD code, synthetic smoke artifacts | B/C | Negative distillation control and artifact-manifest template |
| TGT | Data preparation, three-stage scripts, PCQM data and TGT weights | A/B | Geometry-teacher and artifact-layout reference; no ETKDG import |
| Graphcore GPS++ | IPU code, scaling configs, checkpoints, challenge workflow | A/B | Durable systems/packaging reference; no portable current run |
| MooseML | Deployable GINE + RDKit-descriptor application | C | Streamlit/Docker/logging pattern only |
| Spectral-temporal curriculum | README claim, code tree, no verified paper/checkpoint | C | Negative reproducibility control only |

The central distinction is between **artifact completeness** and **scientific
comparability**. GraphGPT, TGT, and GPS++ are completed systems in their own
roles; none proves a same-contract ETKDG improvement for MolGap. MooseML proves
that an app can be packaged, not that the model is competitive. The spectral
repository supplies neither metric provenance nor a usable baseline.

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
26. **OOF stacking is a delivery protocol, not a free validation boost.** The
     NVIDIA solution gets its ensemble weights from predictions on explicitly
     held-out folds, but those folds are made after combining train and valid
     rows. MolGap can borrow the OOF/Huber mechanics only after candidate roles
     and the sealed evaluation boundary are frozen; a challenge score from a
     39-checkpoint ensemble cannot be compared with one GraphState seed.
27. **A direct-PCQM checkpoint still needs a geometry manifest.** GraphGPT and
    TGT make source and weights easy to retrieve, while TGT separates distance
    prediction from property fine-tuning. Neither public artifact proves the
    project's exact ETKDG construction, atom ordering, or inference path. For
    MolGap, a checkpoint URL must be accompanied by coordinate code, split
    hashes, and a train/inference geometry test.
28. **A completed accelerator system is an engineering reference, not a local
    candidate.** GPS++ demonstrates durable preprocessing, checkpoint, and
    inference boundaries, but its Poplar/IPU custom operations are part of the
    model contract. Portability and score claims cannot be separated from the
    accelerator implementation.
29. **Deployment completeness and scientific completeness are different.**
    MooseML shows a practical SMILES-to-app path with logging and batch
    inference, but its absence of a paper, released checkpoint, and
    independently checkable metric makes it a packaging reference only.
30. **A poor public score can still be useful negative evidence.** The spectral
    curriculum repository's `0.268 eV` claim is not accepted as a benchmark,
    yet its missing split/geometry/checkpoint provenance is a concrete reason
    to reject README-only candidates before compute is spent.
31. **Electronic pretraining needs a local/global attribution split.** GraphQPT
    keeps atom-level quantum pretraining, graph-level PCQM HLG pretraining, and
    masking as separate interventions. A future teacher study should preserve
    that separation and report whether any gain comes from local electronic
    supervision or simply from target-family pretraining.
32. **Distillation must pass a teacher-quality gate before student tuning.**
    The public 3D-prior repository includes the right no-KD control and paired
    intervals, yet its synthetic teacher/student smoke fails to beat the
    baseline. This supports measuring teacher quality and geometry first rather
    than assuming that adding KD losses creates a useful signal.

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
- For the NVIDIA PCQM4Mv2 winner, retain the repository as a delivery-time OOF
  stacking reference. If an ensemble is ever authorized, freeze the candidate
  list first, emit per-fold OOF/test manifests, and fit Huber weights only on
  the permitted internal role; do not reuse its train+valid fold construction
  for a sealed official evaluation.
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
- For GraphGPT, pin the repository revision and ModelScope checkpoint metadata
  before any representation probe; record the Eulerian sequence transform and
  verify whether the selected route can consume the exact ETKDG graph/atom
  ordering. Do not treat README validation values as a current score.
- For TGT, retain the three-stage distance/property separation as a teacher
  design reference. If it is ever audited, hash the HF data/weights, inspect
  the coordinate script, and prove ETKDG equivalence before any use; do not
  import TGT's RDKit/learned-distance geometry into the current cache.
- For Graphcore GPS++, borrow only the durable artifact layout and scaling
  ledger. The IPU/Poplar implementation and checkpoint license must remain
  explicit; do not allocate the bounded A100 screen to an IPU-specific port.
- For MooseML, borrow the deployment smoke-test, batch CSV, and run-log shape
  only. It has no independently verified PCQM metric or released checkpoint.
- For the spectral-temporal curriculum repository, record the README claim as
  a negative reproducibility control and stop; no compute or data retrieval is
  justified without a primary paper, fixed split, checkpoint, and independent
  metric.
- For GraphQPT, retain the atom-level versus graph-level pretraining separation
  and representation diagnostics. If ever revisited, freeze the Zenodo data
  revision and source licenses, then define an ETKDG-only teacher role; do not
  use the PCQM HLG pretraining as an unlabeled auxiliary target.
- For the PCQM 3D-prior distillation repository, borrow its split/geometry
  manifest, no-distillation control, and paired bootstrap layout. Do not infer a
  positive gain from the synthetic smoke; a future real-PCQM audit must use a
  trained teacher-quality gate and keep DFT versus ETKDGv3+UFF explicit.
- Any future use of PCQM SDF must verify the official checksum and ensure that
  validation/test roles remain sealed.

## 10. IEM and MolInteract: interaction-teacher artifacts and the missing-code boundary

### IEM

The [IJCAI paper](https://www.ijcai.org/proceedings/2024/675) and [official
repository](https://github.com/HongxinXiang/IEM) provide the strongest new
image-teacher surface in this continuation. The repository tree exposes
`data_process`, `dataloader`, `distillation`, `loss`, `model`, `pretrain`,
`resumes`, and asset paths. Its README advertises roughly 2M pretraining images
and an `IEM.pth` teacher. This is enough to justify a read-only artifact audit,
not enough to call the teacher retrievable and licensed: an immutable release
hash, complete image manifest, and checkpoint terms were not closed here.

The code-side contract is also important. The visible 3D processing path uses
RDKit embedding and MMFF optimization, so it cannot be copied into MolGap's
ETKDG-only pipeline. IEM is valuable as a layout for frozen teacher outputs,
knowledge-enhancer losses, task-enhancer losses, and graph-only downstream
inference. No repository artifact was downloaded or imported.

### MolInteract

The [PAKDD paper PDF](https://kdd2025.kdd.org/wp-content/uploads/2025/07/paper_22.pdf)
is a method-level source, not a completed code asset in this audit. Targeted
searches for an author-maintained repository, checkpoint, or data release did
not locate a verifiable implementation. The paper still exposes a useful
engineering contract: an eight-layer, approximately 9M-parameter 2D GINE plus
3D SchNet model, repeated interaction layers, and reciprocal relation-prediction
losses trained on PCQM4Mv2. Because the downstream frontier table is QM9 and
the 3D branch is part of the original model, it remains a paper-only method
reference under the present ETKDG/graph-only inference boundary.

### LeJEPA

The [LeJEPA arXiv record](https://arxiv.org/abs/2609.04261) claims release of
code, configurations, and checkpoints, but the audited record did not expose a
repository URL or independently retrievable artifact manifest. Its evaluation
is ogbg-molhiv and antibiotic activity rather than PCQM quantum properties.
The engineering lesson is therefore the control matrix—frozen probe versus
fine-tuning versus scratch—not a weight or dataset import.

## 11. Pretraining alignment, force-centric objectives, and code boundaries

### Does GNN Pretraining Help Molecular Representation?

The [NeurIPS record](https://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html)
and paper provide controlled evidence across objectives, splits, features,
dataset scale, and GNN architecture. The paper is valuable as an evaluation
template, but a verified author-maintained implementation/checkpoint was not
located in this audit. It remains a negative/control source rather than a
downloadable asset.

### ET-OREO

The [NeurIPS ET-OREO record](https://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html)
supports a force-centric equilibrium/off-equilibrium objective and reports a
large multi-corpus conformer pretraining result. Targeted repository and
checkpoint searches did not close an author-maintained implementation or
immutable release. The method uses explicit 3D and force/energy-gradient
roles, so even a future adaptation would need a new ETKDG-only protocol. No
code, data, or weights were downloaded.

### JMP

The [official JMP repository](https://github.com/facebookresearch/JMP) is a
real, inspectable engineering package: it exposes `config`, `src`, `scripts`,
dataset preprocessing, fine-tuning commands, and named `JMP-S`/`JMP-L`
checkpoint links. The README says the repository is archived/deprecated, data
are excluded because of size, and the majority license is CC-BY-NC. This is
enough for a code/configuration audit and license-aware design reference, not
for importing the model into MolGap. The data and checkpoints remain external
and their 3D atomic-property contract is not PCQM/ETKDG Gap.

### CSI / efficient-atom

The [efficient-atom repository](https://github.com/Yasir-Ghunaim/efficient-atom)
is smaller but exposes the full method surface needed to inspect the claim:
structure mapping, upstream/downstream feature extraction, CSI calculation,
individual and mixed pretraining commands, random/class-balanced sampling,
fine-tuning commands, and checkpoint handoff. It depends on the JMP dataset
and checkpoint ecosystem, follows the JMP CC-BY-NC boundary, and does not
contain a PCQM4Mv2 Gap experiment. The code is therefore useful for a future
same-database selector design and equal-budget accounting, but no external
data or checkpoint is an acceptable current MolGap asset.

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
  [code](https://github.com/liuyurou1/DenoiseVAE), [pretrain config](https://raw.githubusercontent.com/liuyurou1/DenoiseVAE/main/config/pretrain_denoisevae.yml),
  [PCQM geometry builder](https://raw.githubusercontent.com/liuyurou1/DenoiseVAE/main/datasets/pcqm4mv2/get_3d_lmdb.py), and
  [model source](https://raw.githubusercontent.com/liuyurou1/DenoiseVAE/main/denoisevae/models/denoise_prednoise.py)
- [Self-Conditioned Denoising paper](https://arxiv.org/html/2603.17196v1), [official code](https://github.com/TyJPerez/SelfConditionedDenoisingAtoms), and [PCQ checkpoint](https://huggingface.co/Ty-Perez/ct-scd-pcq)
- [GraphGPT code and checkpoints](https://github.com/alibaba/graph-gpt)
- [GraphQPT code and curated-data links](https://github.com/aidd-msca/GraphQPT)
- [DGT Nature Communications paper](https://www.nature.com/articles/s41467-026-75005-9),
  [MIT code](https://github.com/zhangsy-ryan/DGT),
  [Zenodo code archive](https://doi.org/10.5281/zenodo.20009509), and
  [Figshare source data](https://doi.org/10.6084/m9.figshare.30665129)
- [QuantumCanvas paper](https://arxiv.org/abs/2512.01519), [MIT code](https://github.com/KurbanIntelligenceLab/QuantumCanvas), and [Zenodo dataset](https://doi.org/10.5281/zenodo.20631934)
- [TMC-Delta-ML paper](https://doi.org/10.1002/chem.71487), [official code](https://github.com/uiocompcat/TMC-Delta-ML), [tmQMg](https://github.com/uiocompcat/tmQMg), and [Zenodo low-fidelity graphs](https://doi.org/10.5281/zenodo.18348669)
- [Selected Machine Learning paper](https://doi.org/10.1039/D2MA00742H), [arXiv record](https://arxiv.org/abs/2110.02596), and [public code](https://github.com/b3rn4rdm/SelectedML)
- [HLP-Stack paper](https://doi.org/10.1039/D5RA08007J), [PMC record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12959570/), and [public repository](https://github.com/college-of-pharmacy-gachon-university/HLP_STACK)
- [PET-MAD-DOS paper](https://doi.org/10.1039/D5DD00557D), [UPET code](https://github.com/lab-cosmo/upet), [Materials Cloud reproduction record](https://doi.org/10.24435/materialscloud:gs-z7), and [Hugging Face model](https://huggingface.co/lab-cosmo/pet-mad-dos)
- [FieldMACE paper](https://doi.org/10.1038/s41524-026-02048-3), [official code](https://github.com/rhyan10/FieldMACE), and [Figshare models/data](https://figshare.com/articles/dataset/Models_data_and_code_for_publication_Incorporating_Long-Range_Interactions_via_the_Multipole_Expansion_into_Ground_and_Excited-State_Molecular_Simulations_/28497857)
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
- [M-OFDFT paper](https://doi.org/10.1038/s43588-024-00605-8), [arXiv PDF](https://arxiv.org/pdf/2309.16578), [Zenodo implementation](https://doi.org/10.5281/zenodo.10616893), and [Figshare model/data collection](https://doi.org/10.6084/m9.figshare.c.6877432)
- [Derivative-aware orbital-free DFT paper](https://doi.org/10.1021/acs.jctc.0c00580) and [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7482319/)
- [QH9 paper](https://arxiv.org/html/2306.09549), [AIRS/QHBench](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9), and [Zenodo data](https://zenodo.org/records/8274793)
- [nablaDFT](https://doi.org/10.1039/D2CP03966D) and [official repository](https://github.com/AIRI-Institute/nablaDFT)
- [∇²DFT paper](https://arxiv.org/html/2406.14347) and [NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2024/file/40d45b1e23d00d5895e65778e85cf8ee-Paper-Datasets_and_Benchmarks_Track.pdf)
- [Self-Consistency Training](https://proceedings.mlr.press/v235/zhang24ak.html), [DEQHNet](https://github.com/Zun-Wang/DEQHNet), and [NeuralSCF](https://github.com/songfeitong/neuralscf)
- [HamEvo paper](https://arxiv.org/html/2606.14498), [official code](https://github.com/axdfhj/HamEvo_official), and [data](https://huggingface.co/datasets/ZJUSCL/hamevo-data)
- [QHFlow](https://arxiv.org/html/2505.18817), [QHFlow2](https://arxiv.org/html/2602.16897), [QHFlow2 code](https://github.com/seongsukim-ml/QHFlow2), and [QH9 checkpoint](https://huggingface.co/ksusu/QHFlow2-QH9)
- [HELM](https://arxiv.org/html/2510.00224), [QHNetV2](https://arxiv.org/html/2506.09398), [SPHNet](https://proceedings.mlr.press/v267/luo25l.html), and [WANet/WALoss](https://arxiv.org/pdf/2502.19227)
- [Fully differentiable ML/QM Hamiltonian learning](https://doi.org/10.1021/acs.jctc.5c00522), [Atomistic Cookbook](https://atomistic-cookbook.org/examples/hamiltonian-qm7/hamiltonian-qm7.html), and [PySCFAD](https://github.com/fishjojo/pyscfad)
- [HamGNN](https://doi.org/10.1038/s41524-023-01130-4), [official code](https://github.com/QuantumLab-ZY/HamGNN), [models](https://doi.org/10.5281/zenodo.8147631), and [training data](https://doi.org/10.5281/zenodo.8157128)
- [HAMSTER](https://doi.org/10.1038/s41467-026-70865-7), [MIT Hamster.jl code](https://github.com/TheoFEM-TUM/Hamster.jl), and [Zenodo data](https://doi.org/10.5281/zenodo.18485403)
- [Graphormer atom-in-a-molecule quantum pretraining paper](https://doi.org/10.1186/s13321-025-00970-0) and [GraphQPT code](https://github.com/aidd-msca/GraphQPT)
- [Tetrahedral Molecular Pretraining paper](https://doi.org/10.1016/j.patcog.2025.112638) and [MIT code/checkpoint repository](https://github.com/sunyuancheng/Tetrahedral-Molecular-Pretraining)
- [PCQM 3D-prior distillation implementation](https://github.com/A-SHOJAEI/pcqm4mv2-3d-prior-distillation)
- [TGT code](https://github.com/shamim-hussain/tgt), [TGT data](https://huggingface.co/datasets/shamim-hussain/pcqm), and [TGT weights](https://huggingface.co/shamim-hussain/tgt)
- [Graphcore GPS++ PCQM implementation](https://github.com/graphcore/ogb-lsc-pcqm4mv2)
- [MooseML HOMO--LUMO Gap application](https://github.com/MooseML/homo-lumo-gap-predictor)
- [Spectral-temporal curriculum repository](https://github.com/A-SHOJAEI/spectral-temporal-curriculum-molecular-gaps)
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
- [NVIDIA Heterogenous Ensemble paper](https://arxiv.org/abs/2211.11035) and [MIT-licensed implementation](https://github.com/jfpuget/NVIDIA-PCQM4Mv2)
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
- [IEM IJCAI paper](https://www.ijcai.org/proceedings/2024/675), [PDF](https://www.ijcai.org/proceedings/2024/0675.pdf), and [official code](https://github.com/HongxinXiang/IEM)
- [MolInteract PAKDD paper PDF](https://kdd2025.kdd.org/wp-content/uploads/2025/07/paper_22.pdf) and [NeurIPS workshop record](https://neurips.cc/virtual/2024/102819)
- [LeJEPA arXiv record](https://arxiv.org/abs/2609.04261)
- [Does GNN Pretraining Help Molecular Representation?](https://papers.nips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html), [paper PDF](https://papers.nips.cc/paper_files/paper/2022/file/4ec360efb3f52643ac43fda570ec0118-Paper-Conference.pdf), and [supplement](https://papers.nips.cc/paper_files/paper/2022/file/4ec360efb3f52643ac43fda570ec0118-Supplemental-Conference.pdf)
- [ET-OREO / May the Force be with You](https://papers.nips.cc/paper_files/paper/2023/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html), [paper PDF](https://papers.nips.cc/paper_files/paper/2023/file/e637029c42aa593850eeebf46616444d-Paper-Conference.pdf), and [OpenReview PDF](https://openreview.net/pdf?id=Ge8Mhggq0z)
- [JMP ICLR record](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4a896a8bb065774169d9ad65f19208b7-Abstract-Conference.html), [paper PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/4a896a8bb065774169d9ad65f19208b7-Paper-Conference.pdf), and [official code](https://github.com/facebookresearch/JMP)
- [CSI paper](https://arxiv.org/abs/2502.11085), [TMLR/OpenReview record](https://openreview.net/forum?id=jfD9BsrDTb), and [official code](https://github.com/Yasir-Ghunaim/efficient-atom)
