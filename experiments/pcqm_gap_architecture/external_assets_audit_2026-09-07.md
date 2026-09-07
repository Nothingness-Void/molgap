# External Code, Weight, Data, and Workflow Audit (2026-09-07)

This is an evidence audit of public assets that may extend the MolGap
knowledge base beyond papers: source repositories, released weights, quantum
chemistry databases, data-ingestion projects, and completed end-to-end
benchmarks. It is separate from the literature records and does not authorize
a model change, a data replacement, a remote job, or an official evaluation.

The audit is anchored to the project contract in
[CURRENT_STATE.md](../../CURRENT_STATE.md),
[ROADMAP.md](../../ROADMAP.md), and the dated PCQM leaderboard audit in
[leaderboard_sota_audit_2026-09-07.md](leaderboard_sota_audit_2026-09-07.md).
Track B selects a Gap-only PCQM4Mv2 specialist from the official-train-derived
100K/10K split. Official validation and test-dev remain sealed, the
architecture screen is fresh random initialization, the parameter target is
bounded, and ETKDG is the train/inference geometry contract.

No external code, dataset, or checkpoint was downloaded or executed during
this audit. “Retrievable” below means that a primary public host exposes the
artifact and its access path; local smoke testing, checksums, and revision
pinning remain mandatory before any artifact can support a MolGap experiment.

## Evidence policy

An item is not admitted to the experiment-eligible pool merely because a
paper reports a good number or a GitHub repository exists. Before a possible
experiment can be written into a protocol, all of the following must be
verified:

1. A primary source identifies the exact code, data, or weight artifact.
2. The artifact is retrievable at the audited revision, rather than only
   promised in a paper or README.
3. The license or usage terms are recorded; ambiguous data rights block use.
4. Target definition, units, level of theory, basis, charge/spin treatment,
   and geometry provenance are known.
5. Molecular identity and role exposure can be checked against PCQM4Mv2,
   PubChemQC, and the other sources already in the repository.
6. The environment, smoke test, artifact hash, and compute envelope are
   reproducible enough to design a bounded acceptance gate.

Evidence grades used below:

| Grade | Meaning | Allowed use |
|---|---|---|
| A | Primary source plus retrievable code/data/weights and enough metadata to reproduce the stated task | Borrowing code or preparing a separately contracted audit; not automatically a MolGap experiment |
| B | Real public artifact, but an important mismatch remains: level, geometry, license, overlap, environment, or task | Background, teacher/OOD candidate after the missing gate is closed |
| C | Scientific claim is useful, but code/data/checkpoints are partial, unavailable, or not independently auditable | Method inspiration only; no compute allocation |
| X | Fails the PCQM comparability or project safety filter | Explicitly exclude from candidate selection |

A grade A means the source project is inspectable or reproducible. It does not
mean the source is scientifically valid for MolGap. A proposed experiment must
also pass the MolGap-specific gates below.

## Executive result

The completed assets with the highest immediate practical value are:

- **OGB evaluator and baseline code** for split, submission, and metric
  hygiene. This is the safest asset to reuse immediately.
- **TGT**, **Uni-Mol+**, **GPS++**, **GraphGPS**, and **GPTrans** for completed
  PCQM implementations with public checkpoints or model configurations. They
  provide upper-bound, inference, and systems evidence, but most exceed the
  current parameter/compute contract or use learned/DFT geometry.
- **QO2Mol** and **OpenQDC** for accessible quantum-data formats and lineage
  ideas. Their methods and geometries are not the MolGap B3LYP/6-31G* target.
- **QCArchive** for durable, queryable, method-explicit quantum calculations.
  It is a workflow/data-generation option, not permission to launch new DFT
  jobs.

The most promising data for a later target-aligned audit are **QCDGE** and the
**NOVA 111,725-molecule collection** because they expose B3LYP/6-31G* HOMO/LUMO
information. Neither is admitted as a training source yet: QCDGE uses a
BJD3-associated ground-state protocol and contains explicit QM9/PubChemQC
subsets; the NOVA portal describes PM6/PM7 geometries and does not establish a
redistribution license on the landing page. Both require identity, geometry,
license, and role audits first.

The most useful domain-specific external checks are the DTU solar database,
NREL/OEDI OPV data, HOPV15, and PhysikMDB. They can test whether errors are
chemically sensible in organic-electronics families, but their methods,
conformers, or experimental values are not a drop-in PCQM training target.

The main negative result is also important: another large graph block is not
the best-supported way to close the gap to the completed PCQM frontier. The
strongest public systems add learned geometry, target-level 3D supervision,
electronic teachers, or much larger multi-stage training. Those are separate
scientific contracts, not free additions to the bounded random-init screen.

### Admission status at audit close

No external model, weight, or database passes the full MolGap experiment gate
yet, because the local smoke test, exact revision/hash, identity-overlap report,
and ETKDG/role decision have not been completed for a named asset. The OGB
evaluator and public source documentation can be reused as engineering and
reading material without changing the scientific screen. Every other item
below remains pending evidence or is explicitly excluded; this audit adds no
scientific experiment to the active queue.

The main database decision is fixed: no source in this audit replaces the
PCQM4Mv2 database used by Track B or the repaired-2M PubChemQC corpus used by
Track A. An external source can be admitted only under an explicitly named
role such as audit, OOD evaluation, or teacher data, with its own identity and
label manifest.

## Completed code and model assets

| Asset | Evidence | What is verifiably available | What MolGap can borrow | Blocking mismatch or risk | Disposition |
|---|---|---|---|---|---|
| [OGB PCQM4Mv2 specification](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) and [official baseline](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/pcqm4m-v2) | A | Official task definition, CID split, evaluator, submission format, GIN/GCN/MLP baselines, and documented 3D-role rule | Evaluator invocation, submission checks, split/role manifest, and baseline sanity tests | Official validation/test-dev coordinates are not available; the project must not import them into the screen | Reuse immediately as evaluation and provenance reference |
| [GraphGPS](https://github.com/rampasek/GraphGPS) | A | MIT code, RWSE/LapPE/SignNet options, GPS configurations, PCQM configs, a downloadable GPS-deep checkpoint, tests, and reported PCQM results | Configuration discipline, positional-encoding tests, batching, and inference/submission flow | README pins an old PyG/PyTorch environment; reported PCQM setup uses a custom validation choice and 6M--19M models, not the MolGap contract | Inspect and borrow patterns; no direct score claim |
| [GPS++ PCQM repository](https://github.com/graphcore/ogb-lsc-pcqm4mv2) | A | MIT code, PCQM pipeline, 11M/22M/44M configurations, released checkpoints, notebooks, and an ensemble result; checkpoints are CC BY 4.0 | Sharded feature preparation, durable training/inference structure, model-zoo metadata, and lightweight scaling references | Optimized for Graphcore IPUs and requires Poplar/custom ops; 44M and the 112-model submission exceed the bounded screen | Completed systems reference; no copy into Track B without a new budget |
| [GPTrans](https://github.com/czczup/GPTrans) | A | Public code, configs, PCQM model zoo links, and reported 6.6M/13.6M/45.7M/86M variants; the 6.6M row reports 0.0833 validation and 0.0842 official test | Edge-to-node/node-to-edge propagation and explicit model/config/checkpoint packaging | Old Python 3.8, PyTorch 1.12, PyG/DGL stack; its edge propagation overlaps the already tested EdgeState family | A completed comparator and code-reading source; no retry of a closed mechanism without a new hypothesis |
| [TGT](https://github.com/shamim-hussain/tgt), [data](https://huggingface.co/datasets/shamim-hussain/pcqm), and [weights](https://huggingface.co/shamim-hussain/tgt) | A | MIT repository, official data-preparation scripts, validation inference, TGT-At and TGT-Agx2 checkpoints; README reports 0.0671 validation/0.0683 test-dev for TGT-At plus RDKit and 0.0682 validation for TGT-Agx2 plus RDKit | Upper-bound teacher, learned distance pipeline, dropout/sample aggregation, and a completed reference for what geometry supervision buys | Three-stage pipeline, learned/DFT geometry, 102M-per-stage TGT-At, PCQM-derived training, and a different conformer contract; using it in the random-init screen risks target/geometry leakage | Highest-value completed teacher/sanity reference; only a separately documented post-selection contract |
| [Uni-Mol](https://github.com/deepmodeling/Uni-Mol) and [Uni-Mol+ instructions/checkpoints](https://github.com/deepmodeling/Uni-Mol/blob/main/unimol_plus/README.md) | A | MIT repository, code, data tools, released checkpoints, PCQM train/valid/test preparation, and small/base/large PCQM models; Uni-Mol+ README reports 27.7M/52.4M/77M models and 0.0714/0.0696/0.0693 validation values | Geometry refinement, multi-conformer aggregation, checkpoint and inference layout, and a teacher representation | Starts from multiple ETKDG+MMFF conformers and refines toward DFT structures; default recipe uses eight GPUs; this is not the single ETKDG input contract | Strong completed geometry reference; not a direct Track B candidate |
| [GPSE](https://github.com/g-taxonomy-workgroup/gpse) and [PCQM4Mv2 Zenodo weight](https://zenodo.org/record/8145095/files/gpse_model_pcqm4mv2_1.0.pt) | B | MIT public positional/structural encoder code and a PCQM4Mv2 pretrained weight are retrievable | Frozen positional/structural teacher or a cheap representation probe after role audit | Pretraining lineage and labels/splits must be checked for overlap; repository records a PyG 2.4.0 incompatibility | Teacher candidate only after provenance and smoke-test gates |
| [Chemprop](https://github.com/chemprop/chemprop) | A | MIT, actively maintained modular MPNN package, tests, examples, CLI, and uncertainty-related interfaces | Ensemble/UQ calibration patterns, CLI/experiment packaging, and independent regression sanity baselines | Not a PCQM-specific completed result and does not establish a Gap gain under this split | Engineering reference; not a scientific Track B candidate |
| [QO2Mol code](https://github.com/kzhoa/QO2Mol) | A/B | Public processing scripts, demo DimeNet++/SphereNet training, v1.3.0 README, Google Drive data, and MD5 checks | Chunked data preparation, checksums, multi-conformer data structures, and data-reader tests | B3LYP/def2-SVP rather than B3LYP/6-31G*; CC BY-NC-SA 4.0; large Google Drive download and version history need freezing | Data-format/teacher reference after license and method audit |
| [OpenQDC](https://github.com/valence-labs/openQDC) and [dataset docs](https://docs.openqdc.io/stable/) | A | Open-source discovery/download layer for many quantum datasets, with dataset-specific metadata and tests | Source lineage, method-aware dataset adapters, checksums, and reproducible downloads | It is a hub, not a common target; each dataset has its own method, geometry, units, and license | Best ingestion-layer pattern; do not merge datasets by filename or SMILES alone |
| [QCArchive](https://docs.qcarchive.molssi.org/) | A | QCFractal/QCPortal/QCFractalCompute platform for querying, submitting, deduplicating, and retrieving quantum records; explicit method/basis/wavefunction options | Durable job/result manifests, find_existing deduplication, method-explicit records, and recoverable result chunks | New DFT computation costs money/time and creates a separate label contract; no computation is authorized by this audit | Future small calibration/label audit only |
| [EDG](https://github.com/HongxinXiang/EDG) | B | IJCAI-25 paper plus MIT code, ImageED checkpoint, 2M ED-feature artifact, ED-aware teacher checkpoint, and downstream teacher features; QM9 table includes HOMO/LUMO/Gap | Three-stage density-image MAE -> structural ResNet18 teacher -> frozen Smooth-L1 geometry-student distillation | EDBench/PCQM-derived 2M source, `6-31G**/+G**` density setup, DFT/source-conformer geometry, OneDrive artifacts, and old PyTorch/PyG/DGL/CUDA stack do not match the B3LYP/6-31G*/ETKDG contract; no direct PCQM Gap result | High-value electronic-teacher pattern only; identity/theory/geometry audit required before any use |
| [EDBench](https://github.com/HongxinXiang/EDBench) | C | Public partial code and paper; repository README still lists full ED data, checkpoints, and several modules as TODO/release items | Electronic retrieval/teacher hypotheses and orbital-energy task decomposition | Full dataset/checkpoints/benchmark code are not all publicly complete; PCQM-derived lineage and basis details require audit | Do not allocate compute; retain as incomplete evidence |
| [HEDMoL](https://github.com/ngs00/HEDMoL) | B | Small public executable repository with training/evaluation scripts and saved result structure | Simple electron-informed substructure teacher and consistency-loss pattern | Reported tasks are experimental molecular properties, source QM9 retrieval can leak evaluation identity, and it is not a PCQM Gap result | Teacher-pattern reference only |
| [EMPP](https://github.com/ajy112/EMPP) | A/B | ICLR 2025 implementation with masking engines, equivariant position-prediction modules, QM9/GEOM entry points, and an auxiliary-loss integration pattern | Masked-position self-supervision and a clean interface for adding a geometry task to an equivariant backbone | Uses a separate 3D coordinate contract; PCQM is used for pretraining and QM9 for reported downstream results; no current PCQM Gap gain | Post-selection objective/code reference; no active initialization |
| [Suiren-1.0](https://github.com/golab-ai/Suiren-Foundation-Model), [fine-tuning code](https://github.com/golab-ai/Suiren-Property-Prediction), and [weights](https://huggingface.co/ajy112/Suiren-Base) | A/B | Public 3D foundation-model code, downstream framework, and model cards/weights; report describes a 1.8B EquiformerV2+MoE teacher and CCD-derived 2D ConfAvg | Frozen-teacher, diffusion-based 3D-to-2D representation distillation and staged adapter design | 70M B3LYP/def2-SVP pretraining report, non-ETKDG geometry, incomplete full-data release, and extreme scale; no PCQM Gap result | High-priority teacher/distillation reference; no direct port |
| [Uni-3DAR paper](https://arxiv.org/abs/2503.16278) and [code](https://github.com/dptech-corp/Uni-3DAR) | B | MIT 3D autoregressive implementation with QM9/DRUG/MP20 pipelines and octree/subtree compression | SpaceFormer 20K HOMO/LUMO/Gap setting and external 19M 3D pretraining corpus; no direct official PCQM4Mv2 Gap result | Hierarchical geometry tokenization and masked next-token engineering | 3D representation reference; no current candidate |
| [ESA paper](https://www.nature.com/articles/s41467-025-60252-z) and [edge-set-attention code](https://github.com/davidbuterez/edge-set-attention) | A | Public edge-set attention implementation, 3D transfer-learning path, and five-run inductive/transductive DFT-to-GW frontier-orbital comparison | Split hygiene, low-/high-fidelity frontier-orbital transfer, and learned attention pooling | QM9/GW targets and explicit 3D coordinates differ from current B3LYP/6-31G*/ETKDG; README documents validation-as-test limitations for its PCQM route | Transfer/delta protocol reference; no current data or checkpoint |
| [MFGP-GEM paper](https://www.nature.com/articles/s41524-024-01479-0) and [code/860-molecule benchmark](https://github.com/ashah1973/MFGP-GEM/tree/main) | A | Public multi-step nonlinear autoregressive GP, dual diffusion-map graph embeddings, and 860 benzoquinones with HF/B3LYP/MP2/CCSD energy/HOMO/LUMO/dipole values | Multi-fidelity hierarchy, direct-vs-delta-vs-autoregressive controls, and high-fidelity count/cost curves | Narrow benzoquinone domain, cc-pVDZ and unrestricted-HF-derived geometry workflow differ from PCQM B3LYP/6-31G*/ETKDG; Python/GPy code is not a PyTorch drop-in | Strong delta/multifidelity protocol reference; no current data or experiment |
| [Multi-fidelity GNN transfer paper](https://www.nature.com/articles/s41467-024-45566-8) and [official code](https://github.com/davidbuterez/multi-fidelity-gnns-for-drug-discovery-and-quantum-mechanics) | A | Six low-to-high transfer strategies, QMugs HOMO/LUMO/quantum-property experiments, transductive/inductive splits, Set Transformer readout, and reproducible QMugs assembly scripts | Proxy labels, predictions, embeddings, pretrain/fine-tune, and readout-only controls; explicit deployment-validity distinction | QMugs GFN2-xTB/DFT, external conformers, and mixed roles are not current B3LYP/6-31G*/ETKDG; transductive low-fi labels may expose test identities | Highest-value multi-fidelity experiment-matrix reference; no current data or initialization |
| [Trainable data-embedding paper](https://doi.org/10.1088/2632-2153/ae0d41) and [VMDatomistic repository](https://github.com/Fraunhofer-SCAI/VMDatomistic) | A/B | Shared M3GNet-like backbone conditioned on trainable dataset/fidelity embeddings; public MultiXC-QM9/MatPES context and evaluation model | Explicitly conditions on functional/basis/data source instead of conflating heterogeneous labels; useful multi-task implementation pattern | Energy/force/atomistic targets, not frontier orbitals; MultiXC-QM9/MatPES and M3GNet geometry contracts differ from current PCQM/ETKDG | Heterogeneous-label and fidelity-conditioning reference; no current target or checkpoint |
| [MoleculeSDE](https://github.com/chao1224/MoleculeSDE), [Geom3D](https://github.com/chao1224/Geom3D), and [Hugging Face checkpoints](https://huggingface.co/chao1224/MoleculeSDE/tree/main) | A/B | MIT code, PCQM4Mv2 pretraining generator, VE/VP 2D-to-3D and 3D-to-2D SDEs, checkpoint index mapped to manuscript tables | Direct data-space denoising objective, local-frame symmetry handling, checkpoint naming, and paired-view manifest design | Published path uses PCQM paired conformations rather than ETKDG; old Python 3.7/PyTorch 1.9/PyG 2.0 stack; no direct PCQM Gap result | High-value pretraining/code reference; no current initialization or run |
| [MoleculeJAE paper](https://arxiv.org/html/2312.03475) | C | NeurIPS paper with trajectory score matching, contrastive surrogate, 2D bond and 3D conformer heads, and detailed QM9 ablations | Noise-trajectory plus original-structure conditioning and a sensitive contrastive-loss ablation | No author-linked executable/checkpoint was verified; QM9 rather than PCQM Gap outcome; paired PCQM geometry is not ETKDG | Method/ablation reference only |
| [MoleBlend](https://github.com/YudiZh/MoleBlend) | B | ICLR paper, public pretraining shell, pretrained-model link, PCQM pretraining instructions, and relation-level SPD/edge/3D blending implementation | Pair-relation alignment before fusion, modality-targeted relation prediction, and noisy-node auxiliary loss | Legacy PyTorch 1.7/PyG 1.6 environment; PCQM Gap is mentioned but no directly checkable score/split table is exposed; geometry is not ETKDG | Relation-level pretraining reference; no current checkpoint or score claim |
| [FlexMol](https://arxiv.org/html/2510.07035) and [official code](https://github.com/tewiSong/FlexMol) | A/B | CIKM 2025 paper and public two-stage code; PCQM SDF checksum, paired encoders/decoders, 2D-only and 3D-only stage commands | Missing-modality completion, separate stage checkpoints, InfoNCE plus reconstruction/consistency losses | Stage 1 uses PCQM paired DFT geometry and Stage 2 uses external Uni-Mol data; no direct PCQM Gap result; full model is 112M parameters | Teacher/missing-modality design reference; no current data or initialization |
| [DenoiseVAE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html) and [code](https://github.com/liuyurou1/DenoiseVAE) | B | ICLR 2025 paper, public two-commit repository, and Appendix A.7 report of PCQM4Mv2 validation `0.0777 +/- 0.0005` with `1.44M` parameters | Molecule-adaptive atom-wise Gaussian noise plus denoising; exact split, coordinate source, checkpoint, and ETKDG inference path remain unverified | Repository page exposes no clear license/checkpoint manifest; DFT equilibrium 3D is not automatically ETKDG | Highest-priority evidence-only pretraining audit; no initialization or run |
| [Pre-training via Denoising](https://arxiv.org/html/2206.00133) and [official code/checkpoint](https://github.com/shehzaidi/pre-training-via-denoising) | A/B | ICLR 2023 paper, MIT repository, documented PCQM4Mv2 pretraining command, `denoised-pcqm4mv2.ckpt`, and QM9 HOMO/LUMO fine-tuning path | Mean-centered equivariant vector-noise prediction; paper separates random init, Noisy Nodes, and upstream PCQM pretraining and reports QM9 Gap transfer | DFT-equilibrium coordinates, full-PCQM upstream role, TorchMD/GNS architecture, and no same-contract PCQM Gap result; released checkpoint cannot be treated as ETKDG-compatible | Completed denoising reference and future ETKDG protocol template; no current initialization or run |
| [Fractional Denoising / Frad](https://arxiv.org/html/2307.10683), [NMI paper](https://arxiv.org/html/2407.11086), [FradNMI code](https://github.com/fengshikun/FradNMI), and [Zenodo models](https://zenodo.org/records/12697467) | A/B | ICML/NMI primary papers, MIT repositories, PCQM4Mv2 pretraining recipe, RN/VRN configurations, retrievable weights, and Figshare source-data record | Hybrid chemical-aware noise changes torsions/bond lengths/angles, final coordinate Gaussian component is the denoising target, and NMI reports QM9/force/robustness ablations | Main PCQM route uses DFT equilibrium geometry; robustness uses RDKit Distance Geometry+MMFF rather than ETKDG; legacy stack, no same-contract PCQM Gap result, and direct torsion route already closed | Complete chemical-aware denoising reference; future ETKDG pretraining audit only, no checkpoint or geometry import |
| [SliDe paper](https://arxiv.org/html/2311.02124) and [MIT code/checkpoints](https://github.com/fengshikun/SliDe) | A/B | ICLR paper, MIT implementation, PCQM4Mv2 label-free pretraining recipe, public QM9/MD17 model links, and force-correlation/ablation tables | Bond/angle/torsion BAT prior with random sliced force estimation; PCQM pretraining uses about 3.4M structures and the paper reports QM9 HOMO/LUMO/Gap and DFT-force audits | Main structures are DFT equilibrium coordinates, the prior is OpenFF/Sage rather than B3LYP, frontier results are QM9 rather than PCQM Gap, and GET/Nv cost is outside the current bounded screen | Strong future ETKDG-only pretraining protocol reference; no current initialization, weight import, or geometry change |
| [CCMD paper](https://arxiv.org/html/2211.16712) | B | Primary paper reports direct PCQM validation for a frozen 3D Graphormer teacher and 2D student, with global/local distillation and size-coordination ablations | Global molecular-token distillation improves the 2D student; coordinated local atom-token loss is needed to avoid the naive local-loss regression; best listed validation is `0.0809` | DFT teacher geometry, prior split wording, large 68M-scale Graphormers, and no verified executable/checkpoint prevent direct reuse under ETKDG | Teacher-loss control template only: no-teacher/global/local/coordinated-local must be a separate future contract |
| [3D-GSRD](https://arxiv.org/abs/2510.16780) and [code](https://github.com/WuChang0124/3D-GSRD) | B | NeurIPS 2025 paper and public code with PCQM pretraining plus QM9 `homo/lumo/gap` fine-tuning scripts | Selective re-mask decoding, 3D relational Transformer, structure-independent decoder | No direct official PCQM Gap number; 3D/DFT geometry and Python 3.8/PyTorch 2.4.1 environment differ from MolGap | Decoder leakage-control reference; no current candidate |
| [3D-MolT5](https://arxiv.org/html/2406.05797) and [code](https://github.com/QizhiPei/3D-MolT5) | B | Apache-2.0 implementation, PCQM 3D pretraining, PubChemQC specialist Gap `0.08`, and 3D ablation `0.0791` vs `0.0968` without 3D | Discrete E3FP-like 3D tokens aligned with SELFIES and text tasks | PubChemQC/text evaluation, external corpora, and non-ETKDG coordinate path; not official OGB PCQM evidence | Compact masked-geometry/reference implementation only |
| [MolSpectra](https://arxiv.org/html/2502.16284) and [code](https://github.com/AzureLeon1/MolSpectra) | B | ICLR 2025 PCQM denoising plus QM9Spectra UV--Vis/IR/Raman pretraining; QM9 Gap `26.8` vs `31.8` meV coordinate baseline | SpecFormer masked spectral patches plus contrastive 3D alignment | QM9S uses B3LYP/def-TZVP and TD-DFT; code page does not expose a clear license/checkpoint; no direct PCQM Gap result | Electronic teacher/auxiliary-objective reference |
| [3D-PGT](https://arxiv.org/html/2306.07812) and [code](https://github.com/LARS-research/3D-PGT) | B | Direct paper PCQM4Mv2 validation `0.0762` with 42.6M parameters; MIT implementation and PCQM command | Automated fusion of bond-length, bond-angle, and dihedral pretexts using total-energy surrogate | DFT 3D pretraining, large GPS model, old PyG 2.0.1, validation-only paper protocol | Completed direct-PCQM pretraining reference; no current initialization |
| [AniDS](https://arxiv.org/html/2510.22123) and [code](https://github.com/ZeroKnighting/AniDS) | B | Public PCQM label-free pretraining command and 4-A100 configuration; downstream force/energy improvements, not Gap | Atom-wise full-covariance anisotropic noise, SO(3)-equivariant covariance correction | 129M Equiformer-scale model, DFT geometry, MD17/OC22 targets, no direct PCQM Gap result | Adaptive-noise design reference |
| [3D-EMGP](https://ojs.aaai.org/index.php/AAAI/article/view/25978) and [code](https://github.com/jiaor17/3D-EMGP) | B | MIT code, checkpoints, energy/force-inspired objective, and QM9 `gap/homo/lumo` fine-tuning | Equivariant denoising plus graph-level noise-scale prediction | GEOM-QM9, old Python/PyTorch/PyG stack, 3D downstream contract, no direct PCQM result | Historical physical-objective reference |
| [Mol-MFFGE paper](https://www.sciencedirect.com/science/article/pii/S0031320325001918) and [code](https://github.com/Yufei-Luo/Mol-MFFGE) | B | Official task-aware pseudo-force-field/meta-learning paper and public preprocessing/training code; QM9 `homo/lumo/delta` task paths | GEOM-QM9/SPICE/MD17, external conformers, no direct PCQM Gap result, and no independently verified checkpoint | Learnable noise transformation plus bi-level auxiliary-loss weighting | Task-aware denoising reference; no current initialization |
| [OCNet](https://www.nature.com/articles/s41524-025-01788-y), [code](https://github.com/545487677/OCNet), and [Zenodo bimolecular assets](https://zenodo.org/records/14934728) | A/B | MIT code, LMDB pretraining datasets, checkpoints, gas-phase HOMO--LUMO Gap scripts, and a CC BY 4.0 bimolecular release; paper reports OCELOT H-L Gap MAE `0.008 eV` | 10M-scale conjugated/dimer pretraining, SE(3) geometry, TB/xTB/DFT descriptors, film/dimer roles | Generated/external chemistry, OCELOT targets, multi-conformer/dimer geometry, and non-PCQM theory; molecular Zenodo release still needs independent retrieval/hash | Highest-value organic-electronics teacher/OOD reference; no current label merge |
| [LUMIA paper](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00713), [MIT code](https://github.com/YajingSun-Group/LUMIA), and [Zenodo data/weights](https://zenodo.org/records/15852302) | A/B | JCTC 2025 paper, MIT repository with pretraining/fine-tuning/explanation/MCTS scripts, and Zenodo archive containing pretraining/downstream data plus model dump with MD5s | ~1.4M organic-molecule pretraining, chemistry-informed edge/substituent masking, RGCN/contrastive objective, OCELOT/optoelectronic downstream roles | No directly verified PCQM Gap result; external organic-optoelectronic target/data/knowledge-mask contract and DGL/RGCN environment | Organic-domain pretraining and interpretability artifact reference; no rows, labels, or weights imported |
| [nablaColors-3D paper](https://www.nature.com/articles/s42004-026-01944-5), [code](https://github.com/AI4DD/nablaColors), and [Zenodo release](https://zenodo.org/records/18061300) | A/B | Public LMDB conformers, scaffold splits, correction manifests, four UniProp checkpoints with MD5s, and explicit RDKit/xTB/DFT geometry paths | Conformer-fidelity benchmark and UniMol+-style low-cost-to-high-fidelity geometry refinement | Experimental optical labels, solvent/CPCM/DFT roles, ETKDG2/MMFF/xTB pipeline, and no PCQM Gap result; not the current ETKDG/B3LYP/6-31G* contract | Completed geometry/artifact reference; no current checkpoint import |
| [DFT-to-experiment frontier-orbital transfer paper](https://www.nature.com/articles/s41524-024-01403-6) | B | Published XGBoost/Klekota--Roth transfer from 11,626 DFT records to 1,198 experimental values; interpretable SHAP fragments | Theory-to-experiment calibration and fragment interpretation | Experimental values are not PCQM Kohn--Sham targets; no current ETKDG/identity contract | External calibration reference only |
| [Conjugated-polymer D-MPNN paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075011/) and [cited repository](https://github.com/Levitsiy/PolymersPropertiesPrediction) | B | Published direct/monomer-DFT/TD-DFT-extrapolated pretraining comparison; Data Availability Statement names model weights/raw training data | Domain-matched teacher selection and full-fine-tuning ablation | Experimental polymer targets, TD-DFT and MMFF94s geometry roles; cited GitHub URL returned `404` during this audit, so no artifact was retrieved | Protocol evidence only; no current data or checkpoint |
| [DFT-feature-assisted optical-gap paper](https://pubs.rsc.org/en/content/articlehtml/2025/nr/d4nr03702b), [SI](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66c23a02f3f4b05290464885/original/si-file.pdf), and [MIT repository](https://github.com/Liu-Group-UF/Machine-Learning-for-Accurate-Optical-Gap-Prediction-in-Conjugated-Polymers) | A/B | Published paper/SI plus public MIT repository with `database-1096`, `new-database-227dp`, and augmentation directories; code/data are visible and retrievable | DFT oligomer feature + ECFP6, nested CV, group interpolation/extrapolation, and teacher-feature ablations | Experimental optical gaps, modified oligomer geometry, B3LYP-D3/6-31G*, and external polymer identity; no PCQM Kohn--Sham result | Completed external protocol/reference; no data or feature merge |
| [Frontier-orbital Chemprop transfer paper](https://pubmed.ncbi.nlm.nih.gov/42268043/) | C | Published article/abstract and publisher preview expose the high-level pretraining/transfer idea and reported errors | Frontier-family transfer and physical-consistency checks | No verified code, checkpoint, split manifest, data identity, or full target-theory packet | Observation only |
| [OPoly26](https://arxiv.org/pdf/2512.23117), [official records](https://huggingface.co/facebook/OMol25), [ColabFit train](https://materials.colabfit.org/id/DS_wfekwbgncjd3_0), and [fairchem](https://github.com/facebookresearch/fairchem) | B | Public paper, Hugging Face/ColabFit records, fairchem implementation path, and stated CC-BY-4.0 release; paper lists multi-million polymer DFT/MD data and frontier properties | Polymer/condensed-phase geometry is not ETKDG; paper/HF specify omegaB97M-V/def2-TZVPD while the ColabFit description says B97M-V/def2-SVP and train metadata advertises only energy/forces; frontier-field coverage is unresolved | Large external asset, theory/version discrepancy, split-wide field availability, and identity overlap require a manifest before any use | External database/schema/OOD audit only; no current data, weights, or teacher |
| [PubChemQC-to-conjugated-oligomer transfer paper](https://pubs.rsc.org/en/content/articlehtml/2025/me/d4me00188e) and [ESI](https://www.rsc.org/suppdata/d4/me/d4me00188e/d4me00188e1.pdf) | B | Published SchNet transfer protocol, public selection criteria/ESI, CO-610 split and direct-vs-transfer metrics; no official code or checkpoint located | PubChemQC-100K is filtered to `106,429` conjugated-like molecules; CO-610 is external B3LYP/6-31G* oligomer data; coordinate construction is not closed to ETKDG | Possible identity overlap with repaired-2M PubChemQC and no reproducible implementation/checkpoint require a source audit before any reuse | Same-source transfer protocol reference; no labels, weights, or rows imported |
| [GFN2-xTB/COCONUT gap workflow](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00186b), [GitHub](https://github.com/sthinius87/HL-gaps-pub), and [Zenodo v0.2.1](https://doi.org/10.5281/zenodo.15113790) | A/B | Published open workflow with `407k` natural products, calculated xTB gaps, descriptors, subgroup analyses, and archived code/data under stated MIT terms | Ten RDKit conformers, xTB/BFGS optimization, and Boltzmann-weighted GFN2-xTB gap are not ETKDG or B3LYP/6-31G*; COCONUT is a separate upstream source | The asset is reproducible but low-fidelity and chemically external; upstream COCONUT terms and exact archive contents still need freezing | Public proxy-generation and delta-cost reference only; no rows, xTB weights, or labels imported |
| [QUED](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00411j), [MIT code/models](https://github.com/lmedranos/QUED), and [Zenodo](https://doi.org/10.5281/zenodo.17106019) | A/B | Peer-reviewed paper, MIT repository, DFTB/CREST scripts, model pickles, HDF5 training sets, and Zenodo archive; QM7-X Gap/descriptor ablations are documented | DFTB3+MBD electronic features and BOB/SLATM representations use QM7-X/PBE0+MBD plus RDKit/MMFF/CREST/GFN2-xTB geometry, not B3LYP/6-31G*/ETKDG | External target/data roles, DFTB+ environment, and feature/weight provenance must stay separate from PCQM; no exact-label overlap audit has been done | Strong electronic-teacher packaging reference; no descriptors, rows, or weights imported |
| [POS-EGNN/OMol25 paper](https://pubs.rsc.org/en/Content/ArticleLanding/2026/EB/D6EB00024J), [IBM/materials](https://github.com/ibm/materials), and [Hugging Face model](https://huggingface.co/ibm-research/materials.pos-egnn) | B | Peer-reviewed OMol25 electrolyte pretraining paper; Apache-2.0 POS-EGNN code, example notebook, and public `pos-egnn.v1-6M.pt` MPtrj weights | Paper: >20M OMol25 structures, `ωB97M-V/def2-TZVPD`, explicit MD solvation/ion-pair geometry, HOMO/LUMO/Gap/site-charge heads; public weights: 1.4M MPtrj energy/force/stress pretraining | No paper-specific frontier checkpoint, no PCQM/B3LYP/6-31G*/ETKDG result, and no permission to transfer external geometry/labels into current roles | Multi-task physical-consistency and public 3D-foundation reference; no rows, labels, or weights imported |
| [AEGCNN-MTL paper](https://www.nature.com/articles/s41524-025-01917-7) | B | Peer-reviewed QM9 HOMO/LUMO/Gap multi-task tables and explicit single-task/multi-task and weak-task negative-transfer comparisons | QM9 and private BDG data; no public code, no public BDG release, no PCQM4Mv2/ETKDG path | Result is a task-grouping reference, not a retrievable implementation or current target result | No data/code import; retain only as multi-task control evidence |
| [OSCs_RGGN](https://github.com/AsadKhanJBNU/OSCs_RGGN) | C | Repository claims a 48,182-sample OSC dataset and RGNN predictions | Domain-specific residual-gated graph baseline | No auditable paper-linked table, dataset archive, license, split, target theory, or checkpoint | Quarantine discovery lead; no data or compute |
| [GLACIER paper](https://arxiv.org/html/2606.11382), [code](https://github.com/eemokey/glacier), and [checkpoint](https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol) | B | Public KDD 2026 multimodal student--teacher implementation, MIT code, and loadable checkpoint; graph/SMILES/descriptor fusion with MiniMol/MolFormer teachers | Enamine 100K pretraining, TDC/MoleculeNet evaluation, no PCQM Gap result, and external teacher/source overlap | Frozen teacher-embedding and multi-teacher ablation pattern after architecture selection | Engineering teacher reference; no current initialization |
| [ChemBERTa-3 paper](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00348b), [code](https://github.com/deepforestsci/chemberta3), and [Zenodo](https://zenodo.org/records/18235841) | B | Public open foundation-model training/benchmark framework, model variants, data preparation, and MIT repository | ZINC20/PubChem pretraining, heterogeneous model sizes, and no directly verified PCQM Gap result | Environment/config/benchmark manifest and model-scale accounting | Engineering reference; no external data or checkpoint import |
| [ChemFM paper](https://www.nature.com/articles/s42004-025-01793-8), [code](https://github.com/TheLuoFengLab/ChemFM), and [Zenodo](https://zenodo.org/records/17450883) | B/C | Public causal SMILES foundation-model study with roughly 3B parameters, 178M UniChem pretraining, and scaling experiments | No matched PCQM Gap result, no ETKDG path, and incompatible scale/source role | Corpus/version/parameter scaling ledger only | Foundation reference; no current candidate |
| [CondPSE](https://arxiv.org/abs/2607.25169) | B | Paper reports polynomial-filtered structural encoder and explicit synthetic/molecular comparison with GPSE | No official implementation or direct PCQM Gap result; molecular-property gain is not consistent | Negative control for structural-pretraining admission | Do not allocate compute |
| [GCPE publisher page](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-026-60014-0) | C | Publisher abstract describes spatial/spectral/subgraph positional encoding and PCQM4Mv2 experiments | Exact metric, split, code, checkpoint, and complete accessible paper not independently available | Require a full table and inference contract before any use | Observation only |
| [Chemprop benchmark v2](https://github.com/chemprop/chemprop_benchmark_v2) and [Zenodo data](https://zenodo.org/records/10078142) | B | MIT benchmark repository with `qm9_gap`/`pcqm4mv2`, explicit `data.csv`/`splits.json`, scripts, checkpoints, and Chemprop v2.0.3 environment | Baseline/engineering package, not a new Gap method or replacement split | Data-loader, metric, split, and artifact-retention sanity reference | No run authorized |
| [KD scalability paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202503271) and [official code](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties) | B | Open-access QM9 teacher/student study with SchNet/DimeNet++/TensorNet, downloadable models, L1+cosine KD, uncertainty weighting, and Optuna | QM9 teacher includes HOMO/LUMO/Gap, but no PCQM/ETKDG result; Python 3.9 and external datasets | Teacher/no-teacher, student-capacity, and embedding-alignment control template | Teacher-loss reference; no current data or checkpoint import |
| [ECMMR publisher page](https://www.sciencedirect.com/science/article/pii/S0957417426009103) | C | Abstract-level claim for BRICS hypergraphs, cross-modal latent tasks, PCQM pretraining, and 22 tasks | Exact metrics, code, checkpoint, and split not exposed in the audited source | Index only until primary artifacts are available | Quarantine |

The values in this table are source-project facts. They are not estimates of a
MolGap improvement. In particular, the TGT and Uni-Mol+ numbers cannot be
compared directly with the 100K/10K GraphState number because data scale,
geometry role, pretraining, and evaluation role differ.

## Public quantum and organic-electronics data

| Source | Evidence and target facts | Geometry / method issue | Overlap, license, or role issue | Safe use before a new contract |
|---|---|---|---|---|
| [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) | 3.7M-scale OGB benchmark; DFT HOMO--LUMO Gap in eV; CID-based 90/2/4/4 split; CC BY 4.0 | Training has DFT-equilibrium 3D, validation/test-dev do not expose explicit 3D | Official roles are sealed by project policy; keep architecture selection train-derived | Existing benchmark only |
| [PubChemQC B3LYP/6-31G*//PM6 collection](https://nakatamaho.riken.jp/pubchemqc.riken.jp/b3lyp_pm6_datasets.html), [project page](https://nakatamaho.riken.jp/pubchemqc.riken.jp/) | Official site lists an 86M B3LYP/6-31G* dataset, downloadable raw/JSON/PostgreSQL forms; the dataset paper describes HOMO/LUMO and Gap-scale electronic data | B3LYP single points are associated with PM6 geometries, not the PCQM equilibrium/ETKDG input path | Likely molecular-lineage overlap with PubChem/PCQM; large raw files and PostgreSQL/JSON versions must be version-pinned; page states CC BY 4.0 for the PM6 datasets | Data-lineage/identity audit, train-role teacher or scale study only; never silently treat PM6 geometry as ETKDG labels |
| [PubChemQC-100K -> CO-610 transfer study](https://pubs.rsc.org/en/content/articlehtml/2025/me/d4me00188e) and [public ESI](https://www.rsc.org/suppdata/d4/me/d4me00188e/d4me00188e1.pdf) | Paper reports a filtered `106,429`-molecule PubChemQC pretraining pool and `610` external B3LYP/6-31G* conjugated oligomers with HOMO/LUMO/Gap; ESI exposes the selection rule and candidate data | The paper does not close the oligomer coordinate-generation method to ETKDG; the pretraining pool is a subset of the same PubChemQC lineage | Exact identities may overlap repaired-2M/Track A; no code/checkpoint; ESI data and paper license/role must be frozen before reuse | Same-source transfer protocol and identity-audit reference; no current data or weights |
| [GFN2-xTB/COCONUT gap workflow](https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00186b) and [Zenodo v0.2.1](https://doi.org/10.5281/zenodo.15113790) | Paper/record expose a complete `407k`-molecule low-fidelity xTB gap/descriptors asset and reproducible ten-conformer/Boltzmann aggregation workflow | GFN2-xTB is a proxy target; RDKit/xTB geometries violate the present ETKDG-only input contract | MIT code/data terms and upstream COCONUT provenance must remain separate; natural-product rows are not current PCQM roles | CPU proxy/delta audit reference after exact ETKDG adaptation and residual/cost measurement; no database merge |
| [QCDGE](https://langroup.site/QCDGE/) and [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362161/) | 443,106 C/N/O/F molecules; ground-state geometries/frequencies and HOMO/LUMO among 27 properties; B3LYP/6-31G* with BJD3; downloadable CSV/HDF5/subsets/checksum/extraction materials | Optimized ground-state geometry and BJD3 protocol are not identical to PCQM's target geometry contract | Includes QM9/GDB-11 and PubChemQC-derived subsets; explicit identity deduplication is mandatory; the named [GitHub link](https://github.com/Yifei-Zhu/Database_codes) currently does not resolve, so only the dataset portal/paper is evidence for code availability | High-value external/teacher source after schema, license, identity, and role audit; not yet a training source |
| [GW frontier-orbital dataset](https://www.nature.com/articles/s41597-023-02486-4) and [Figshare archive](https://figshare.com/articles/dataset/Accurate_GW_frontier_orbital_energies_of_134_kilo_molecules_of_the_QM9_dataset_/21610077) | 133,885 QM9 molecules; public PBE/G0W0/GW@PBE HOMO/LUMO records with 132,151 default GW convergences and basis-set metadata | GW quasiparticle energies and CP2K GAPW/aug-cc basis protocol are not B3LYP/6-31G* Kohn-Sham PCQM labels | QM9 identity overlap and different theory/geometry require exact audit; no gap field identical to the current target | External high-fidelity teacher/delta reference only; never append to current DB |
| [VQM24 paper](https://www.nature.com/articles/s41597-025-05428-4), [Zenodo data](https://zenodo.org/records/15442257), and [code](https://github.com/dkhan42/VQM24) | 835,947 converged structures; 784,875 minima, 51,072 saddles, 258,242 isomers, 577,705 conformers, 5,599 stoichiometries; orbital energies, charges, multipoles, wavefunctions, and a DMC subset | GFN2-xTB/CREST and \omegaB97X-D3/cc-pVDZ/PSI4, plus DMC PBE0/cc-pVQZ, are not ETKDG or B3LYP/6-31G*; mixtures of stationary points and conformers matter | Large wavefunction archive, nine-element/5-heavy-atom closed-shell coverage, and distinct license/identity/conformer roles require a manifest; no PCQM-equivalent label contract | External coverage/OOD benchmark or teacher-design reference only; no database merge |
| [QCML paper](https://www.nature.com/articles/s41597-025-04720-7), [Zenodo record](https://zenodo.org/records/14859804), and [public TFDS bucket](https://console.cloud.google.com/storage/browser/qcml-datasets/tfds/) | 17.2M graphs, 14.7B GFN0/GFN2 examples, 33.5M PBE0 examples, off-equilibrium conformers, forces, multipoles, orbital quantities, matrices, charges/spins, and explicit outlier/status fields | Open Babel/UFF/GFN0-xTB/conformer search/normal-mode geometry and PBE0/FHI-aims numeric basis are not ETKDG+B3LYP/6-31G*; targets are energy/force/electronic matrices, not current Gap labels | Multi-terabyte public assets and broad elements/states require selective manifests; no current-label identity contract | Schema, outlier, low/high-fidelity transfer, and cost-accounting reference; no bulk merge or current pretraining |
| [qcMol paper](https://www.nature.com/articles/s42004-026-02076-6), [server](https://structpred.life.tsinghua.edu.cn/qcmol/), and [code](https://github.com/GHUSER-haoyu/qcMol) | 1,200,216 curated molecules; 31 quantum descriptors, including global Gap and atom/bond electronic descriptors; 247,448 scaffolds; B3LYP-D3/def2-SV(P)//GFN2-xTB | Single xTB-optimized geometry, D3/basis mismatch, mixed source datasets, and charged/open-shell fractions differ from PCQM/ETKDG | Canonical/source overlap and data/parameter license must be frozen before use; experimental ADMET labels are a separate role | Strong external electronic-teacher or representation audit candidate after identity and theory filtering; no database merge |
| [QM40 paper](https://www.nature.com/articles/s41597-024-04206-y), [Figshare data](https://doi.org/10.6084/m9.figshare.25993060.v1), and [code](https://github.com/Ayeshmadu/QM40_dataset_for_ML) | 162,954 neutral singlet ZINC drug-like molecules, 10--40 atoms; HOMO/LUMO/HL_gap, coordinates, Mulliken charges, and local bond-strength descriptors at B3LYP/6-31G(2df,p) | DFT/xTB geometry and 6-31G(2df,p) differ from ETKDG and B3LYP/6-31G*; size/elements and charged-state policy differ | Paper/repository permission statements differ; exact data release/license and PCQM overlap require audit | Near-target external size/scaffold stress test or teacher reference only; no concatenation |
| [QeMFi paper](https://www.nature.com/articles/s41597-024-04247-3), [Zenodo data](https://doi.org/10.5281/zenodo.13925688), and [code](https://github.com/vivinvinod/QeMFi) | 135,000 geometries from 9 molecules, five CAM-B3LYP/ORCA basis fidelities, excitation/dipole/SCF properties, and per-fidelity CPU times with MFML/o-MFML scripts | Wigner/geodesic excited-state geometries and TD-DFT excitation targets are not the current ETKDG/B3LYP Gap contract | Benchmark is intentionally narrow; time/cost results are not general PCQM gains | Delta/multifidelity protocol and cost-accounting reference only |
| [MFΔML paper](https://arxiv.org/html/2410.11391) and [MFDeltaML code](https://github.com/SM4DA/MFDeltaML) | Public scripts for Δ-ML, MFML, o-MFML, MFΔML, and predicted-baseline controls, with QeMFi learning/cost curves | KRR/Coulomb-matrix models, five basis-set fidelities, and nine molecules are not the current neural PCQM/ETKDG contract | The code README points to Zenodo snapshot 12734761 while the QeMFi paper uses 13925688; release must be pinned before reproduction | Delta cost/proxy and leakage protocol reference; no current code or data import |
| [ViSNetGWBSE paper](https://pubs.rsc.org/en/content/articlehtml/2026/sc/d5sc09780k), [official code](https://github.com/daoiradrio/ViSNetGWBSE), and [QM9GWBSE target data](https://zenodo.org/records/17902233) | Public ViSNet pretrain/fine-tune code, checkpoints, test data, OMol25/QCDGE low-fidelity sources, and 133,885-molecule qsGW/qsGW-BSE release; paper compares full versus readout-only transfer and 10k--120k target-data learning curves | omegaB97M-V/def2-TZVPD, omegaB97X-D/6-31G(d), qsGW/GW-BSE, and explicit Cartesian geometries are not B3LYP/6-31G* Kohn--Sham Gap on the ETKDG contract | External-source identity/role overlap, target-theory alignment, and checkpoint provenance must be frozen; public code defaults to a 120k/10k/3,885 target split and a frozen representation in transfer mode | Multi-fidelity transfer protocol and outlier/data-demand reference; no current database merge or pretraining |
| [NOVA 111,725 collection](https://novaresearch.unl.pt/en/datasets/energies-of-the-homo-and-lumo-orbitals-for-111725-organic-molecul/) and [Figshare DOI](https://doi.org/10.6084/m9.figshare.3384184.v1) | University repository identifies 111,725 organic molecules and HOMO/LUMO energies at B3LYP/6-31G*//PM6 or //PM7; DOI and publication date are recorded | PM6/PM7 geometry provenance is explicit; it is not automatically an ETKDG or PCQM equilibrium set | Landing page does not establish a redistribution license; file schema and canonical identities have not yet been independently checked | Candidate for an identity-controlled external validation or teacher audit only after license/file verification |
| [PhysikMDB](https://physikmdb.uni-graz.at/) and [Python package](https://pypi.org/project/physikmdb/) | Live searchable molecular-orbital database exposes XC, basis, Gap, IP/IE, EA, code/version, author/date; package is GPL-3.0-only and supports offline saves | Database is heterogeneous; visible records include B3LYP/cc-pVTZ, so B3LYP/6-31G* must be explicitly filtered and counted | Public query/API rate limits and per-record download behavior need a cached, cited snapshot; GPL applies to the package, not necessarily every database record | Small targeted conjugation/oligomer sanity set, never bulk training by default |
| [QMugs](https://www.nature.com/articles/s41597-022-01390-7) and [OpenQDC QMugs docs](https://docs.openqdc.io/stable/API/datasets/qmugs.html) | Hundreds of thousands of drug-like molecules and millions of conformers with orbital properties; QMugs V2 documentation describes a PM6 extension | GFN2-xTB or PM6 geometry and different DFT/basis choices; not exact PCQM target | Method/version metadata must be preserved; wavefunction data is very large | Multi-fidelity/geometry teacher and OOD benchmark, not exact-label augmentation |
| [QO2Mol v1.3.0](https://github.com/kzhoa/QO2Mol) | README states over 20M configurations, ten elements, B3LYP/def2-SVP properties, and MD5 checks; paper describes 120K molecules and 20M conformers | def2-SVP and many conformers are not B3LYP/6-31G*/ETKDG; conformer is an explicit data identity | CC BY-NC-SA 4.0; download and historical version must be frozen before use | Geometry diversity/teacher/OOD only |
| [OPoly26](https://arxiv.org/pdf/2512.23117), [ColabFit train](https://materials.colabfit.org/id/DS_wfekwbgncjd3_0), and [OPoly26 validation schema](https://huggingface.co/datasets/colabfit/OPoly26-val) | Public polymer database with `>6.35M` DFT calculations, `>1.2B` atoms, and paper-listed HOMO/HOMO--LUMO-gap properties; public train/validation surfaces are retrievable | Paper/HF specify omegaB97M-V/def2-TZVPD, while the ColabFit description says B97M-V/def2-SVP; condensed-phase/MD/DFTB/AFIR geometry is not ETKDG | Train record currently advertises energy/forces, validation schema exposes `electronic_band_gap`, and split-wide frontier coverage/identity overlap remain unresolved | External polymer OOD/schema audit or separately contracted pretraining reference; no merge or current teacher |
| [THEMol](https://github.com/ByteDance-Seed/THEMol) and [Hugging Face data](https://huggingface.co/datasets/ByteDance-Seed/THEMol) | Released code/data describe more than 3B DFT calculations, Hessians, relaxation trajectories, torsion scans, and MBIS multipoles; validation checks hashes and HDF5/CSV consistency | It supplies forces, Hessians, torsion/relaxation information, not HOMO/LUMO/Gap labels in the MolGap format | Code Apache-2.0; data CC BY-NC 4.0; non-commercial data restriction must be respected | Geometry/torsion/electronic-descriptor teacher after architecture selection; not a current target source |
| [QM7-X](https://www.nature.com/articles/s41597-021-00812-2), [VQM24](https://github.com/dkhan42/VQM24), and [QO2Mol](https://arxiv.org/abs/2410.19316) | Finished public small-molecule quantum datasets with multiple conformers and electronic/wavefunction-related properties | PBE0+MBD, DMC/PBE0, or B3LYP/def2-SVP methods differ from the target | Each has distinct data licenses and duplicate/identity concerns | Secondary geometry/electronic teacher references only |
| [DTU donor--acceptor database](https://cmr.fysik.dtu.dk/solar/solar.html) | Domain-specific database reports B3LYP Kohn--Sham HOMO/LUMO/Gap and related solar-cell quantities | Not the PCQM split or necessarily the same optimized/conformer protocol | Small, domain-specific database; do not infer broad accuracy from it | External organic-electronics chemistry-family check |
| [NREL/OEDI OPV database](https://data.openei.org/submissions/8285) | Public OPV collection with canonical SMILES, optimized structures, basis/functional metadata, HOMO/LUMO/Gap and optical quantities | Mixed DFT functionals/bases and 3D structures; not a single target level | Keep method/basis per row; license applies to the published collection, not arbitrary derived labels | OOD/error-stratification set, not training labels |
| [HOPV15](https://pmc.ncbi.nlm.nih.gov/articles/PMC5037972/) | 350 organic molecules/polymers with multiple low-energy conformers and experimental/calculated OPV properties | Experimental and calculated values are mixed and not a single B3LYP/6-31G* target | Very small and domain-specific | Qualitative external stress test for conjugated systems |
| [Curated DFT HOMO--LUMO set](https://computational.cancer.gov/dataset/curated-dft-homo-lumo) | 346,780 compounds curated with duplicate removal using chemical identifiers and DFT gaps | Underlying methods are heterogeneous and not the MolGap target | Its deduplication/curation procedure is useful; target data cannot be merged blindly | Borrow curation checks and use as method-diverse OOD context |

## What is worth borrowing

### 1. Borrow data-contract discipline before borrowing a model

The highest-confidence improvement is an asset manifest, not another neural
block. Every imported row or teacher output should carry:

- canonical identity and source identity: CID, InChIKey, canonical SMILES, and
  conformer identity when applicable;
- target name, unit, sign convention, charge, spin, functional, basis, code,
  version, and convergence/status fields;
- geometry source and construction method, including ETKDG, PM6, DFT-relaxed,
  GFN2-xTB, MMFF, or learned;
- source release/version, file checksum, code revision, license, and permitted
  role: train, teacher, external, or quarantine;
- overlap status against PCQM4Mv2 official-train CIDs and all existing
  PubChemQC/Track A identity manifests.

This reuses the strongest operational ideas in OGB, OpenQDC, QCDGE, QO2Mol, and
QCArchive. It fits the existing MolGap data_repair.py, experiment_db.py,
artifact hashes, and acceptance-record architecture. It should be implemented
as a reusable capability only when a specific source is admitted; it is not
permission to merge sources now.

### 2. Use completed PCQM projects as teachers and upper bounds, not as claims

TGT and Uni-Mol+ are the most valuable completed references because their
repositories expose code, data preparation, checkpoints, and inference paths.
Their results establish that learned geometry, distance supervision, multiple
conformers, and staged training can materially change the PCQM score. They do
not prove that a small random-init GraphState should be modified in any
particular way.

A later teacher experiment would need a separate contract specifying whether:

- the teacher is queried only on MolGap ETKDG inputs;
- teacher checkpoints trained on PCQM data are allowed only as a train-role
  teacher and never evaluated on sealed roles;
- the student sees teacher predictions, frozen embeddings, or geometry targets;
- teacher-derived fields are stored with hashes and no target-role leakage; and
- the experiment is compared to a fresh matched GraphState control.

Until those choices are frozen, downloading a checkpoint is an audit action,
not a scientific experiment.

### 3. Prefer target-aligned electronic data over larger mismatched data

QCDGE and NOVA are more scientifically relevant to the B3LYP/6-31G* target than
large force/energy collections. Their first use should be a read-only audit:
canonicalize identities, compute Gap from supplied HOMO/LUMO fields, inspect
geometry and basis metadata, and quantify overlap. If overlap is high, those
rows cannot serve as an independent test set. If source role and license are
acceptable, a small teacher/auxiliary-label study could be considered after
the PCQM architecture is selected.

PubChemQC's large B3LYP/6-31G* collection is valuable for scale and label
lineage, but its PM6 geometry makes it a different input/label distribution.
It should not be described as an ETKDG-consistent extension without a measured
contract and matched inference path.

### 4. Add a real external chemistry-family evaluation

The DTU solar, NREL/OEDI OPV, HOPV15, and filtered PhysikMDB records can expose
failure modes in long conjugation, donor--acceptor separation, oligomer length,
charge/spin, and method sensitivity. They should be reported with per-row
method/basis/conformer metadata and grouped by chemistry family. They cannot
replace official PCQM validation or be pooled into training as if all gaps had
the same definition.

### 5. Borrow geometry and electronic descriptors only after a contract change

Uni-Mol+, QO2Mol, OPoly26, QMugs, THEMol, EDG, and HEDMoL provide evidence for geometry
refinement, conformer ensembles, MBIS/charge/electron-density teachers, or
substructure-level transfer. None is a free input feature under the present
ETKDG/random-init screen. A future geometry or teacher route should choose one
source, preserve its method and license, and compare it to a fresh same-budget
control.

## Candidate experiment gate

A source may enter a future protocol only after the coordinator has a compact
candidate card containing:

1. primary URLs, repository commit/tag, dataset release, and exact checkpoint;
2. retrieval result, file size, SHA-256/MD5, and a local or remote smoke test;
3. code/data/checkpoint licenses and any non-commercial or share-alike limits;
4. target, level, basis, geometry method, units, charge/spin, and convergence
   definitions;
5. canonical identity and conformer deduplication report against every
   relevant MolGap role;
6. explicit role map proving that no official validation/test-dev data or
   labels enter architecture selection;
7. ETKDG train/inference decision, or an explicit new geometry contract with
   user authorization;
8. parameter count, memory, throughput, wall-clock estimate, and durable
   checkpoint/output plan;
9. a fresh matched control, one material hypothesis, and a quantitative stop
   gate before any seed expansion;
10. a dated protocol and acceptance script in the owning experiment directory.

If any item is unknown, the source remains B/C evidence and is not a possible
remote experiment. A README assertion, a search-result snippet, or a paper
headline cannot fill a missing item.

## Explicit exclusions

- **MoiréGT/RadialFocus PCQM numbers** remain excluded from the 2D leaderboard
  comparison because their reported PCQM route uses physical 3D coordinates;
  see the leaderboard audit.
- **DeMol and TetraGT full reproductions** are not bounded candidates: the
  reported systems are much larger and, for TetraGT, the public repository
  still states that code is coming soon.
- **Mixed-method databases** such as OPV collections, QMugs, QM7-X, and the
  curated multi-source HOMO/LUMO set cannot become exact B3LYP/6-31G* labels by
  renaming a column.
- **PCQM-derived checkpoints or electron-density teachers** cannot enter the
  fresh random-init architecture screen. They require a separate pretraining,
  distillation, or teacher contract with role and leakage accounting.
- **NOVA data** cannot be redistributed or used in a training artifact until
  the actual file, license, and identity mapping are verified.
- **EDBench** is not treated as a completed reproducibility target while its
  repository still lists core data/checkpoint/code releases as incomplete.

## Audit boundary and follow-up

This record adds no code, no downloaded model, no dataset merge, no remote job,
and no change to the active Track B queue. The safe follow-up is an
evidence-only manifest/overlap audit for one named source at a time. A
scientific experiment begins only after the candidate-card gate passes and a
dated protocol is accepted under the repository's existing remote durability,
resource-separation, seed-budget, and ETKDG rules.

## Primary sources

- [OGB PCQM4Mv2 specification](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
- [OGB PCQM baseline code](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/pcqm4m-v2)
- [GraphGPS](https://github.com/rampasek/GraphGPS)
- [GPS++](https://github.com/graphcore/ogb-lsc-pcqm4mv2)
- [GPTrans](https://github.com/czczup/GPTrans)
- [TGT code](https://github.com/shamim-hussain/tgt), [TGT data](https://huggingface.co/datasets/shamim-hussain/pcqm), [TGT weights](https://huggingface.co/shamim-hussain/tgt)
- [Uni-Mol](https://github.com/deepmodeling/Uni-Mol), [Uni-Mol+ PCQM instructions](https://github.com/deepmodeling/Uni-Mol/blob/main/unimol_plus/README.md)
- [GPSE](https://github.com/g-taxonomy-workgroup/gpse), [Zenodo checkpoint](https://zenodo.org/record/8145095/files/gpse_model_pcqm4mv2_1.0.pt)
- [Chemprop](https://github.com/chemprop/chemprop)
- [QO2Mol code and data instructions](https://github.com/kzhoa/QO2Mol), [QO2Mol paper](https://arxiv.org/abs/2410.19316)
- [OpenQDC](https://github.com/valence-labs/openQDC), [OpenQDC documentation](https://docs.openqdc.io/stable/)
- [QCArchive documentation](https://docs.qcarchive.molssi.org/)
- [PubChemQC project](https://nakatamaho.riken.jp/pubchemqc.riken.jp/), [B3LYP/6-31G*//PM6 dataset page](https://nakatamaho.riken.jp/pubchemqc.riken.jp/b3lyp_pm6_datasets.html), [dataset paper](https://arxiv.org/abs/2305.18454)
- [QCDGE portal](https://langroup.site/QCDGE/), [QCDGE dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362161/)
- [NOVA 111,725-molecule record](https://novaresearch.unl.pt/en/datasets/energies-of-the-homo-and-lumo-orbitals-for-111725-organic-molecul/)
- [PhysikMDB](https://physikmdb.uni-graz.at/), [PhysikMDB Python package](https://pypi.org/project/physikmdb/)
- [QMugs article](https://www.nature.com/articles/s41597-022-01390-7), [QMugs OpenQDC documentation](https://docs.openqdc.io/stable/API/datasets/qmugs.html)
- [THEMol code](https://github.com/ByteDance-Seed/THEMol), [THEMol data](https://huggingface.co/datasets/ByteDance-Seed/THEMol)
- [EDG](https://github.com/HongxinXiang/EDG), [EDBench](https://github.com/HongxinXiang/EDBench), [HEDMoL](https://github.com/ngs00/HEDMoL), [EMPP](https://github.com/ajy112/EMPP)
- [Suiren-1.0 report](https://arxiv.org/abs/2603.21942), [foundation code](https://github.com/golab-ai/Suiren-Foundation-Model), [property code](https://github.com/golab-ai/Suiren-Property-Prediction), and [Suiren-Base weights](https://huggingface.co/ajy112/Suiren-Base)
- [Uni-3DAR paper](https://arxiv.org/abs/2503.16278) and [official code](https://github.com/dptech-corp/Uni-3DAR)
- [GW frontier-orbital paper](https://www.nature.com/articles/s41597-023-02486-4), [arXiv record](https://arxiv.org/abs/2303.08708), and [Figshare dataset](https://figshare.com/articles/dataset/Accurate_GW_frontier_orbital_energies_of_134_kilo_molecules_of_the_QM9_dataset_/21610077)
- [VQM24 paper](https://www.nature.com/articles/s41597-025-05428-4), [arXiv record](https://arxiv.org/abs/2405.05961), [Zenodo data](https://zenodo.org/records/15442257), [VQM24 code](https://github.com/dkhan42/VQM24), and [OpenQDC adapter](https://docs.openqdc.io/stable/API/datasets/vqm24.html)
- [QCML paper](https://www.nature.com/articles/s41597-025-04720-7), [Zenodo record](https://zenodo.org/records/14859804), and [TFDS bucket](https://console.cloud.google.com/storage/browser/qcml-datasets/tfds/)
- [qcMol paper](https://www.nature.com/articles/s42004-026-02076-6), [server](https://structpred.life.tsinghua.edu.cn/qcmol/), [code](https://github.com/GHUSER-haoyu/qcMol), and [parameter archive](https://zenodo.org/records/19183364)
- [QM40 paper](https://www.nature.com/articles/s41597-024-04206-y), [Figshare record](https://doi.org/10.6084/m9.figshare.25993060.v1), and [code](https://github.com/Ayeshmadu/QM40_dataset_for_ML)
- [QeMFi paper](https://www.nature.com/articles/s41597-024-04247-3), [Zenodo data](https://doi.org/10.5281/zenodo.13925688), and [code](https://github.com/vivinvinod/QeMFi)
- [MFΔML paper](https://arxiv.org/html/2410.11391), [MFDeltaML code](https://github.com/SM4DA/MFDeltaML), and [QeMFi snapshot](https://zenodo.org/records/12734761)
- [MoleculeSDE paper](https://arxiv.org/html/2305.18407), [official code](https://github.com/chao1224/MoleculeSDE), [Geom3D](https://github.com/chao1224/Geom3D), and [checkpoints](https://huggingface.co/chao1224/MoleculeSDE/tree/main)
- [MoleculeJAE paper](https://arxiv.org/html/2312.03475) and [NeurIPS record](https://papers.neurips.cc/paper_files/paper/2023/hash/acddda9cd6f310689f7657f947705a99-Abstract-Conference.html)
- [MoleBlend paper](https://arxiv.org/html/2307.06235) and [official code](https://github.com/YudiZh/MoleBlend)
- [FlexMol paper](https://arxiv.org/html/2510.07035) and [official code](https://github.com/tewiSong/FlexMol)
- [DenoiseVAE paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/37e9e62294ff6607f6f7c170cc993f2c-Abstract-Conference.html), [paper PDF](https://openreview.net/attachment?id=ym7pr83XQr&name=pdf), and [code](https://github.com/liuyurou1/DenoiseVAE)
- [Frad ICML paper](https://arxiv.org/html/2307.10683), [Frad code](https://github.com/fengshikun/Frad), [Frad NMI paper](https://arxiv.org/html/2407.11086), [FradNMI code](https://github.com/fengshikun/FradNMI), [Zenodo weights](https://zenodo.org/records/12697467), and [source data](https://doi.org/10.6084/m9.figshare.25902679.v1)
- [SliDe paper](https://arxiv.org/html/2311.02124) and [MIT code/checkpoints](https://github.com/fengshikun/SliDe)
- [CCMD paper](https://arxiv.org/html/2211.16712)
- [3D-GSRD paper](https://arxiv.org/abs/2510.16780) and [official code](https://github.com/WuChang0124/3D-GSRD)
- [3D-MolT5 paper](https://arxiv.org/html/2406.05797) and [official code](https://github.com/QizhiPei/3D-MolT5)
- [MolSpectra paper](https://arxiv.org/html/2502.16284) and [official code](https://github.com/AzureLeon1/MolSpectra)
- [3D-PGT paper](https://arxiv.org/html/2306.07812) and [official code](https://github.com/LARS-research/3D-PGT)
- [AniDS paper](https://arxiv.org/html/2510.22123) and [official code](https://github.com/ZeroKnighting/AniDS)
- [3D-EMGP paper](https://ojs.aaai.org/index.php/AAAI/article/view/25978) and [official code](https://github.com/jiaor17/3D-EMGP)
- [Mol-MFFGE paper](https://www.sciencedirect.com/science/article/pii/S0031320325001918) and [official code](https://github.com/Yufei-Luo/Mol-MFFGE)
- [OCNet paper](https://www.nature.com/articles/s41524-025-01788-y), [official code](https://github.com/545487677/OCNet), [molecular Zenodo record](https://zenodo.org/records/14935486), and [bimolecular Zenodo record](https://zenodo.org/records/14934728)
- [DFT-to-experiment frontier-orbital transfer paper](https://www.nature.com/articles/s41524-024-01403-6)
- [LUMIA paper](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00713), [MIT code](https://github.com/YajingSun-Group/LUMIA), and [Zenodo data/weights](https://zenodo.org/records/15852302)
- [Conjugated-polymer D-MPNN paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13075011/), [cited repository](https://github.com/Levitsiy/PolymersPropertiesPrediction)
- [DFT-feature-assisted optical-gap paper](https://pubs.rsc.org/en/content/articlehtml/2025/nr/d4nr03702b), [SI](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66c23a02f3f4b05290464885/original/si-file.pdf), and [MIT repository](https://github.com/Liu-Group-UF/Machine-Learning-for-Accurate-Optical-Gap-Prediction-in-Conjugated-Polymers)
- [Frontier-orbital Chemprop transfer paper](https://pubmed.ncbi.nlm.nih.gov/42268043/), [DOI](https://doi.org/10.1063/5.0333521)
- [OPoly26 paper](https://arxiv.org/pdf/2512.23117), [official OMol25 page](https://huggingface.co/facebook/OMol25), [ColabFit train record](https://materials.colabfit.org/id/DS_wfekwbgncjd3_0), [OPoly26 validation schema](https://huggingface.co/datasets/colabfit/OPoly26-val), and [fairchem code](https://github.com/facebookresearch/fairchem)
- [PubChemQC-to-conjugated-oligomer transfer paper](https://pubs.rsc.org/en/content/articlehtml/2025/me/d4me00188e) and [supporting information](https://www.rsc.org/suppdata/d4/me/d4me00188e/d4me00188e1.pdf)
- [OSCs_RGGN repository](https://github.com/AsadKhanJBNU/OSCs_RGGN)
- [nablaColors-3D paper](https://www.nature.com/articles/s42004-026-01944-5), [code](https://github.com/AI4DD/nablaColors), and [Zenodo release](https://zenodo.org/records/18061300)
- [POS-EGNN/OMol25 paper](https://pubs.rsc.org/en/Content/ArticleLanding/2026/EB/D6EB00024J), [IBM/materials](https://github.com/ibm/materials), and [Hugging Face model](https://huggingface.co/ibm-research/materials.pos-egnn)
- [AEGCNN-MTL paper](https://www.nature.com/articles/s41524-025-01917-7)
- [GLACIER paper](https://arxiv.org/html/2606.11382), [official code](https://github.com/eemokey/glacier), and [checkpoint](https://huggingface.co/glacier-hf/GLACIER-100k-MiniMol)
- [ChemBERTa-3 paper](https://pubs.rsc.org/en/content/articlelanding/2026/dd/d5dd00348b), [code](https://github.com/deepforestsci/chemberta3), and [Zenodo release](https://zenodo.org/records/18235841)
- [ChemFM paper](https://www.nature.com/articles/s42004-025-01793-8), [code](https://github.com/TheLuoFengLab/ChemFM), and [Zenodo](https://zenodo.org/records/17450883)
- [GPSE paper](https://arxiv.org/html/2307.07107), [code](https://github.com/G-Taxonomy-Workgroup/GPSE), and [PCQM-named checkpoint](https://zenodo.org/record/8145095/files/gpse_model_pcqm4mv2_1.0.pt)
- [CondPSE](https://arxiv.org/abs/2607.25169), [GCPE publisher page](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-026-60014-0)
- [Chemprop benchmark v2](https://github.com/chemprop/chemprop_benchmark_v2) and [Zenodo data](https://zenodo.org/records/10078142)
- [ECMMR publisher page](https://www.sciencedirect.com/science/article/pii/S0957417426009103)
- [KD scalability paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202503271) and [official code](https://github.com/PEESEgroup/Knowledge-Distillation-For-Molecular-Properties)
- [ViSNetGWBSE paper](https://pubs.rsc.org/en/content/articlehtml/2026/sc/d5sc09780k), [official code](https://github.com/daoiradrio/ViSNetGWBSE), and [QM9GWBSE target data](https://zenodo.org/records/17902233)
- [ESA frontier-transfer paper](https://www.nature.com/articles/s41467-025-60252-z) and [official code](https://github.com/davidbuterez/edge-set-attention)
- [DTU solar database](https://cmr.fysik.dtu.dk/solar/solar.html), [NREL/OEDI OPV data](https://data.openei.org/submissions/8285), [HOPV15](https://pmc.ncbi.nlm.nih.gov/articles/PMC5037972/)
- [QM7-X](https://www.nature.com/articles/s41597-021-00812-2), [VQM24](https://github.com/dkhan42/VQM24), [curated DFT HOMO-LUMO set](https://computational.cancer.gov/dataset/curated-dft-homo-lumo)
