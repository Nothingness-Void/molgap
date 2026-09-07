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
| [EDG](https://github.com/HongxinXiang/EDG) | B | Public IJCAI implementation with teacher/student electron-density image machinery and requirements | Cross-modal teacher/student and distillation design | Old PyTorch/PyG/DGL/CUDA stack; electron-density image data and QM9/rMD17 task, not PCQM Gap | Pattern only; no direct port |
| [EDBench](https://github.com/HongxinXiang/EDBench) | C | Public partial code and paper; repository README still lists full ED data, checkpoints, and several modules as TODO/release items | Electronic retrieval/teacher hypotheses and orbital-energy task decomposition | Full dataset/checkpoints/benchmark code are not all publicly complete; PCQM-derived lineage and basis details require audit | Do not allocate compute; retain as incomplete evidence |
| [HEDMoL](https://github.com/ngs00/HEDMoL) | B | Small public executable repository with training/evaluation scripts and saved result structure | Simple electron-informed substructure teacher and consistency-loss pattern | Reported tasks are experimental molecular properties, source QM9 retrieval can leak evaluation identity, and it is not a PCQM Gap result | Teacher-pattern reference only |

The values in this table are source-project facts. They are not estimates of a
MolGap improvement. In particular, the TGT and Uni-Mol+ numbers cannot be
compared directly with the 100K/10K GraphState number because data scale,
geometry role, pretraining, and evaluation role differ.

## Public quantum and organic-electronics data

| Source | Evidence and target facts | Geometry / method issue | Overlap, license, or role issue | Safe use before a new contract |
|---|---|---|---|---|
| [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) | 3.7M-scale OGB benchmark; DFT HOMO--LUMO Gap in eV; CID-based 90/2/4/4 split; CC BY 4.0 | Training has DFT-equilibrium 3D, validation/test-dev do not expose explicit 3D | Official roles are sealed by project policy; keep architecture selection train-derived | Existing benchmark only |
| [PubChemQC B3LYP/6-31G*//PM6 collection](https://nakatamaho.riken.jp/pubchemqc.riken.jp/b3lyp_pm6_datasets.html), [project page](https://nakatamaho.riken.jp/pubchemqc.riken.jp/) | Official site lists an 86M B3LYP/6-31G* dataset, downloadable raw/JSON/PostgreSQL forms; the dataset paper describes HOMO/LUMO and Gap-scale electronic data | B3LYP single points are associated with PM6 geometries, not the PCQM equilibrium/ETKDG input path | Likely molecular-lineage overlap with PubChem/PCQM; large raw files and PostgreSQL/JSON versions must be version-pinned; page states CC BY 4.0 for the PM6 datasets | Data-lineage/identity audit, train-role teacher or scale study only; never silently treat PM6 geometry as ETKDG labels |
| [QCDGE](https://langroup.site/QCDGE/) and [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11362161/) | 443,106 C/N/O/F molecules; ground-state geometries/frequencies and HOMO/LUMO among 27 properties; B3LYP/6-31G* with BJD3; downloadable CSV/HDF5/subsets/checksum/extraction materials | Optimized ground-state geometry and BJD3 protocol are not identical to PCQM's target geometry contract | Includes QM9/GDB-11 and PubChemQC-derived subsets; explicit identity deduplication is mandatory; the named [GitHub link](https://github.com/Yifei-Zhu/Database_codes) currently does not resolve, so only the dataset portal/paper is evidence for code availability | High-value external/teacher source after schema, license, identity, and role audit; not yet a training source |
| [NOVA 111,725 collection](https://novaresearch.unl.pt/en/datasets/energies-of-the-homo-and-lumo-orbitals-for-111725-organic-molecul/) and [Figshare DOI](https://doi.org/10.6084/m9.figshare.3384184.v1) | University repository identifies 111,725 organic molecules and HOMO/LUMO energies at B3LYP/6-31G*//PM6 or //PM7; DOI and publication date are recorded | PM6/PM7 geometry provenance is explicit; it is not automatically an ETKDG or PCQM equilibrium set | Landing page does not establish a redistribution license; file schema and canonical identities have not yet been independently checked | Candidate for an identity-controlled external validation or teacher audit only after license/file verification |
| [PhysikMDB](https://physikmdb.uni-graz.at/) and [Python package](https://pypi.org/project/physikmdb/) | Live searchable molecular-orbital database exposes XC, basis, Gap, IP/IE, EA, code/version, author/date; package is GPL-3.0-only and supports offline saves | Database is heterogeneous; visible records include B3LYP/cc-pVTZ, so B3LYP/6-31G* must be explicitly filtered and counted | Public query/API rate limits and per-record download behavior need a cached, cited snapshot; GPL applies to the package, not necessarily every database record | Small targeted conjugation/oligomer sanity set, never bulk training by default |
| [QMugs](https://www.nature.com/articles/s41597-022-01390-7) and [OpenQDC QMugs docs](https://docs.openqdc.io/stable/API/datasets/qmugs.html) | Hundreds of thousands of drug-like molecules and millions of conformers with orbital properties; QMugs V2 documentation describes a PM6 extension | GFN2-xTB or PM6 geometry and different DFT/basis choices; not exact PCQM target | Method/version metadata must be preserved; wavefunction data is very large | Multi-fidelity/geometry teacher and OOD benchmark, not exact-label augmentation |
| [QO2Mol v1.3.0](https://github.com/kzhoa/QO2Mol) | README states over 20M configurations, ten elements, B3LYP/def2-SVP properties, and MD5 checks; paper describes 120K molecules and 20M conformers | def2-SVP and many conformers are not B3LYP/6-31G*/ETKDG; conformer is an explicit data identity | CC BY-NC-SA 4.0; download and historical version must be frozen before use | Geometry diversity/teacher/OOD only |
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

Uni-Mol+, QO2Mol, QMugs, THEMol, EDG, and HEDMoL provide evidence for geometry
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
- [EDG](https://github.com/HongxinXiang/EDG), [EDBench](https://github.com/HongxinXiang/EDBench), [HEDMoL](https://github.com/ngs00/HEDMoL)
- [DTU solar database](https://cmr.fysik.dtu.dk/solar/solar.html), [NREL/OEDI OPV data](https://data.openei.org/submissions/8285), [HOPV15](https://pmc.ncbi.nlm.nih.gov/articles/PMC5037972/)
- [QM7-X](https://www.nature.com/articles/s41597-021-00812-2), [VQM24](https://github.com/dkhan42/VQM24), [curated DFT HOMO-LUMO set](https://computational.cancer.gov/dataset/curated-dft-homo-lumo)
