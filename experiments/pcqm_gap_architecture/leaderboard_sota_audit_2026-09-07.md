# PCQM4Mv2 Leaderboard and Frontier Audit (2026-09-07)

This is a dated external benchmark audit for the Track B decision. It answers
four questions that a normal literature list cannot answer reliably:

1. Which numbers are official OGB submissions and which are paper-only
   validation numbers?
2. What information, scale, pretraining, and geometry contract produced the
   leading numbers?
3. How far is the current MolGap anchor from those numbers, with the caveat
   that the contracts are not identical?
4. Which work is worth running, and which work should be rejected before it
   consumes a long remote training allocation?

The base 73-source ledger and the incremental source reserve remain the
reading records:
[`recent_literature_coverage_ledger_50.md`](recent_literature_coverage_ledger_50.md)
and [`literature_extension_2026.md`](literature_extension_2026.md). This file
owns the leaderboard/comparability audit only. It does not change
`CURRENT_STATE.md`, `ROADMAP.md`, a frozen decision, or any remote-run
authorization.

## Executive verdict

- The current public **official OGB submission board** is led by the TGT entry
  `TGT-At(+RDKit)` at `0.0683 eV` test-dev MAE. The OGB page labels this row
  `EGT+Tri. Attn.+RDKit Coords.`; the TGT paper uses the TGT name.
- The current **paper-reported validation frontier** is lower but is not an
  official test-dev result. DeMol reports `0.0603 eV` after pretraining and
  no additional fine-tuning; TetraGT reports `0.0659 eV` with RDKit and
  `0.0671 eV` without RDKit. These are large, multi-stage systems, not
  directly comparable to the current bounded random-init screen.
- A search result that appears even better, MoiréGT/RadialFocus at about
  `0.0463 eV`, is not admitted to the PCQM ranking: the source explicitly
  evaluates PCQM as a 3D graph task with physical coordinates available. That
  conflicts with the OGB validation/test inference contract and is exactly the
  kind of attractive but non-comparable number the audit is meant to catch.
- MolGap's frozen GraphState anchor is `0.1297801534 eV` on an
  official-train-derived `100K/10K` internal split, with `3.6658M` parameters,
  fresh random initialization, and ETKDG-consistent geometry. A naive numeric
  comparison is `0.0615 eV` and about `1.90x` the official TGT test-dev MAE;
  this is context, not an official leaderboard gap because data role, training
  scale, and model contract differ.
- Therefore the current route is **not world-leading in absolute PCQM
  accuracy**. It is a defensible latest-feasible route under the project's
  strict `<=4M`, single-A100/12-hour, random-init, ETKDG, and sealed-role
  constraints. The active directed-bond and SignNet-LapPE screens should be
  understood as low-cost information-flow tests, not as attempts to reproduce
  the global SOTA.
- The main missing factor is not another untested local topology primitive. The
  leading PCQM systems combine learned or iteratively refined geometry,
  DFT-geometry/electronic supervision, multi-stage pretraining, substantially
  larger encoders, and sometimes stochastic or ensemble inference. Copying
  those recipes would change the MolGap contract and, in several cases, would
  violate the hard ETKDG train/inference consistency rule. That boundary must
  be made explicit before any leaderboard-parity project is started.

## Benchmark contract and ranking hygiene

The official [OGB PCQM4Mv2 specification](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/)
defines graph regression of the DFT HOMO--LUMO gap in eV with MAE. There are
`3,378,606` training molecules with DFT-equilibrium 3D structures; validation
and test molecules do not provide explicit 3D coordinates. The split is
`90/2/4/4` by PubChem CID into train/validation/test-dev/test-challenge. The
community may use training 3D information, but test-time inference is required
without explicit test 3D coordinates.

This creates three different evidence classes:

| Evidence class | What it means | Use in MolGap |
|---|---|---|
| OGB submitted leaderboard | Organizer-evaluated test-dev number, with declared hardware and submission metadata | Highest-confidence external ranking |
| Paper validation result | Public-label validation result, often after pretraining, with the authors' own protocol | Mechanism and upper-bound context; never call it test-dev |
| Community aggregate | Automatically extracted mixture of validation and test-dev values | Discovery index only; verify against the primary paper |

The [official OGB-LSC leaderboard](https://ogb.stanford.edu/docs/lsc/leaderboards/)
currently lists the TGT-family submission as its best PCQM4Mv2 test-dev row.
The [community Graph Learning Benchmarks page](https://graphlearningbenchmarks.github.io/datasets/pcqm4mv2/)
currently places DeMol first at `0.0603`, but the page itself is an automatic
aggregate that warns it may contain mistakes and mixes paper validation with
official test-dev values. It is not an organizer leaderboard. TetraGT's
published result is also absent from the community top table, which is a
practical reminder that the aggregate is not exhaustive.

## Official OGB submitted board

The following is a normalized reading of the rows currently shown by OGB. The
first model name preserves the paper naming where the OGB label is historical.
Validation and test-dev are both shown because a large validation/test-dev
disagreement is itself useful evidence about comparability.

| OGB order | Model / OGB label | Val MAE (eV) | Test-dev MAE (eV) | Params | Declared hardware | Interpretation |
|---:|---|---:|---:|---:|---|---|
| 1 | TGT-At(+RDKit) / `EGT+Tri. Attn.+RDKit Coords.` | 0.0671 | **0.0683** | 203.9M | 32 NVIDIA V100 32GB | Learned distance/conformer route; optional RDKit initialization |
| 2 | TGT-At / pure neural | 0.0686 | 0.0698 | 203.9M | 32 NVIDIA V100 32GB | Same family without RDKit coordinates |
| 3 | Uni-Mol+(+RDKit) | 0.0693 | 0.0705 | 77.0M | 8 A100 | Eight ETKDG+MMFF conformers, iterative refinement |
| 4 | Uni-Mol+ base | 0.0696 | 0.0708 | 52.4M | 8 A100 | Smaller Uni-Mol+ variant |
| 5 | GraphGPT-L48 | 0.0682 | 0.0709 | 810M | 16 L40S | Much larger graph model |
| 6 | GraphGPT | 0.0700 | 0.0717 | 455.9M | 8 L40S | Much larger graph model |
| 7 | GPS++ ensemble | 0.0778 | 0.0720 | 44.3M | Graphcore Bow Pod16 | Ensemble; not a single-model comparison |
| 8 | MolNet ensemble | 0.0797 | 0.0753 | 32.0M | 8 RTX 3090 | Ensemble |
| 9 | Global-ViSNet | 0.0784 | 0.0766 | 78.45M | 4 A100 | Geometry/pretraining family |
| 10 | Transformer-M | 0.0772 | 0.0782 | 68.96M | 4 A100 | Large 2D/3D pretraining-era baseline |

Useful size anchors from the same official page are `GPTrans-T` at
`0.0833 eV` validation / `0.0842 eV` test-dev with `6.58M` parameters and
`HFAGNN` at `0.1005 eV` validation / `0.1010 eV` test-dev with `3.95M`
parameters. The latter is the closest public parameter-scale reference, but
it still uses a different full-data training recipe and hardware.

## Paper-reported 2026 frontier

### DeMol: lowest reported validation number, not an official submission

[DeMol (ICLR 2026)](https://arxiv.org/abs/2603.00568) reports `0.0603 eV`
on the PCQM4Mv2 validation set after pretraining, without additional
fine-tuning. Its full system is about `186M` parameters and uses parallel
atom-centric and bond-centric graphs, cross-level atom--bond attention,
torsional encoding, covalent-radius regularization, and structure-aware
attention masks. The reported recipe uses 12 layers, width 768 streams, 128
Gaussian kernels, multi-task pretraining on the PCQM training set, and about
seven days on eight A6000 GPUs.

The number is scientifically important but has three limits for MolGap:

1. It is validation, not an OGB test-dev submission.
2. The model learns from PCQM DFT 3D structures during pretraining, while
   MolGap's current architecture screen is fresh random-init and
   ETKDG-consistent.
3. The official DeMol repository is not available in the paper record; the
   result is therefore not an inexpensive reproducibility target.

The strongest transferable evidence is the ablation: the gain is associated
with a real bond-centric stream and atom--bond interaction, then refined by
geometry/electronic auxiliary objectives. MolGap already tested a bounded
atom--bond adaptation and closed it; this does not justify repeating the
full-width model under the current budget.

### TetraGT: newest explicit higher-order geometry result

[TetraGT (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/239b0f62a2cb86876a0c7028393d2a18-Paper-Conference.pdf)
reports the following validation results:

| Variant | Params | PCQM validation MAE | Geometry/input contract |
|---|---:|---:|---|
| TetraGT-6 | 60M | 0.0693 eV | Three-stage conformer/denoising pipeline |
| TetraGT-12 | 127M | 0.0681 eV | Same |
| TetraGT-24 | 215M | 0.0671 eV | Learned geometry from 2D, no RDKit initialization |
| TetraGT-24(+RDKit) | 215M | **0.0659 eV** | RDKit initialization plus learned refinement |

Its novelty is explicit bond-angle and torsion-angle tokens, tetrahedral
interaction, directed cyclic angle loss, and hierarchical virtual nodes. The
paper reports about 34 A100 GPU-days for the full PCQM route. The authors'
[public repository](https://github.com/xkxxfyf/TetraGT) currently says the code
is still coming soon. Thus TetraGT is a useful frontier mechanism reference,
but it is not a ready-to-run MolGap candidate and its smallest model is already
about 15 times the current parameter ceiling.

### TGT and Uni-Mol+: the reproducible shape of the official frontier

[TGT](https://arxiv.org/html/2402.04538) uses a three-stage route: predict a
pairwise distance structure from the 2D graph, train a task predictor with noisy
DFT 3D structures and distance supervision, then fine-tune using predicted
distances. Stochastic dropout inference aggregates multiple predictions. The
paper reports about 32 A100 GPU-days for the 203M-parameter model; the
RDKit-initialized official submission reaches `0.0683 eV` test-dev.

[Uni-Mol+](https://www.nature.com/articles/s41467-024-51321-w) starts from
eight ETKDG+MMFF conformers, trains iterative refinement toward DFT
equilibrium structures, and averages the eight predictions at inference. Its
77M-parameter variant reaches `0.0705 eV` test-dev and `0.0693 eV` validation.
The ablation indicates that iterative refinement and target-conformation
supervision, rather than a single larger atom encoder, are the important
ingredients.

These two families show why the official score is difficult to obtain under a
small random-init budget: geometry is not just an input feature; it is a
learned intermediate target and a separate training stage.

### The 46 meV false lead: MoiréGT/RadialFocus

[MoiréGT](https://openreview.net/pdf?id=sJzfxRbEv6) reports `46.3 meV`
validation and `46.4 meV` test-dev on PCQM4Mv2 with an eight-layer,
512-dimensional, 16-head model. [RadialFocus](https://doi.org/10.1145/3746252.3760877)
reports the same `46.3 meV` validation headline with about 13M parameters.
These numbers must not be used as a target for the current project. The
MoiréGT experiment section says it evaluates **3D graphs**, computes distances
from physical locations, and then reports the PCQM result. The official OGB
specification says explicit 3D coordinates are unavailable for validation and
test inference. The RadialFocus abstract likewise describes its PCQM result as
a 3D molecular benchmark but does not supply a valid official submission record.

Disposition: retain these papers as evidence that distance-modulated attention
can be effective when coordinates are genuinely available; exclude their PCQM
numbers from every 2D leaderboard and from the MolGap candidate pool until an
independent, coordinate-free, official-role reproduction exists. This is a
hard comparability filter, not a judgment about whether the architecture works
on a legitimate 3D task.

## Track A production is a different benchmark

The shipped [Track A decision](../../production/04_evaluate/project_freeze/track_a_final_decision.md)
and the Track B PCQM specialist must not be compared as if they were one
leaderboard entry. Track A selected the repaired-2M dense
2D ensemble at `0.097638 eV` on a fixed `1,973`-molecule external B3LYP
common set, with OOD and hard-subset reporting. Its fixed `4,981`-row
PCQM4Mv2 validation proxy was `0.302120 eV`, and the project explicitly
accepted that result because the production objective is a broader B3LYP
distribution rather than the Gap-only PCQM specialist. Track B's `0.129780 eV`
is instead a 100K/10K internal architecture-selection result. Neither number
is an official PCQM test-dev score, and neither should be used to claim a
cross-track SOTA ranking.

## Frontier outside the direct PCQM leaderboard

The current molecular-ML frontier is also moving toward better electronic
teachers and larger foundation data. These are relevant to the long-term
MolGap reserve but are not direct replacements for PCQM4Mv2 labels.

| Work | Frontier contribution | Why it is not a direct PCQM score |
|---|---|---|
| [DGT, Nature Communications 2026](https://www.nature.com/articles/s41467-026-75005-9) | Separate atom/bond Transformers with mutual fusion, ring/RPE/RWPE/stereogeometric features; 3D improves QM9 frontier-orbital errors substantially | No direct official PCQM Gap leaderboard result; several experiments use 3D or small PCQM pretraining subsets |
| [EDBench, NeurIPS 2025](https://arxiv.org/abs/2505.09262) | 3.36M PCQM-derived electron densities plus orbital energies, energy components and multipoles | Closest electronic teacher, but its basis differs from PCQM and its released splits/lineage require exact role-overlap auditing |
| [ED-DiT, August 2026](https://arxiv.org/abs/2608.03260) | Masked diffusion pretraining on electron density with electron-number consistency; improves orbital-energy transfer under limited labels | No direct PCQM Gap result; inherits EDBench data availability and leakage questions |
| [ELECTRA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/288b63aa98084366c4536ba0574a0f22-Paper-Conference.pdf) | Floating Gaussian-orbital representation for charge-density reconstruction and faster DFT initialization | Charge-density/SCF objective, not HOMO-LUMO Gap; 3D electronic representation |
| [qcMol, Communications Chemistry 2026](https://www.nature.com/articles/s42004-026-02076-6) | 1.2M molecules, 31 DFT properties, local atom/bond electronic descriptors and broad chemistry | B3LYP-D3/def2-SV(P)//GFN2-xTB, not PCQM B3LYP/6-31G*; single optimized geometry |
| [OMol25 / UMA](https://arxiv.org/abs/2505.08762) | Over 100M DFT calculations, broad elements, charges/spins and large-scale universal molecular modeling | Primarily energy/force/electronic-structure foundation data, not a matched HOMO-LUMO Gap target |
| [Machine learning for frontier orbital energetics review](https://doi.org/10.1016/j.cartre.2026.100674) | 59-study review; equivariant models have lower reported median error than invariant models, but confounding is substantial | Cross-paper metrics, data, pretraining and target levels are not aligned; reporting of cost and uncertainty is incomplete |
| HEDMoL, MET, Q-GEM, QCDGE and bond/charge/OP teacher lines | Transfer electronic substructure, atom-level quantum descriptors, charges and bond-order information | Teacher datasets require overlap, level-of-theory and leakage audits before use with PCQM |

The practical message is that an “electronic teacher” is now a more credible
new information source than another arbitrary contact edge. The existing
[`literature_extension_2026.md`](literature_extension_2026.md) records the
source-level routes and their target/level-of-theory warnings.

## MolGap gap analysis

The local comparator is not an official validation row. It is the accepted
three-seed GraphState result on exactly `100,000` train-role and `10,000`
internal-validation graphs:

| Reference | MAE (eV) | Params / scale | Naive difference from GraphState | Naive ratio | What the comparison is allowed to say |
|---|---:|---|---:|---:|---|
| MolGap GraphState9 mean | **0.129780** | 3.6658M; 100K/10K; 3 seeds | — | 1.00x | Valid local selection anchor |
| HFAGNN official validation | 0.1005 | 3.95M; official full train | 0.0293 | 1.29x | Closest public compact context; not an equal-data rerun |
| GPTrans-T official validation | 0.0833 | 6.58M; official full train | 0.0465 | 1.56x | Public mid-size context; hardware/training differ |
| TGT-At(+RDKit) official test-dev | 0.0683 | 203.9M; 32 V100 | 0.0615 | 1.90x | Official external target, not an equal-contract delta |
| TGT-At(+RDKit) paper validation | 0.0671 | 203.9M; multi-stage | 0.0627 | 1.93x | Paper/official-family context |
| TetraGT-24 pure 2D paper validation | 0.0671 | 215M; multi-stage | 0.0627 | 1.93x | 2D inference claim, but not small-model or same-training contract |
| TetraGT-24(+RDKit) paper validation | 0.0659 | 215M; multi-stage | 0.0639 | 1.97x | Geometry-refinement context |
| DeMol paper validation | 0.0603 | 186M; pretraining | 0.0695 | 2.15x | Lowest reported paper number, not official test-dev |

Relative to GraphState, reaching the compact HFAGNN context would require a
roughly `22.6%` MAE reduction; reaching TGT/TetraGT would require roughly
`48%`; reaching DeMol would require roughly `53.5%`. These percentages are
planning diagnostics only. They must not be converted into claims that the
100K/10K GraphState model has been fairly benchmarked against full-data models.

## Is the current run using the latest research?

The answer depends on which meaning of “latest” is intended:

| Meaning | Answer | Reason |
|---|---|---|
| Lowest absolute PCQM number | **No** | DeMol/TetraGT paper validation and TGT official test-dev are substantially lower |
| Latest *feasible* bounded discovery under MolGap rules | **Yes, narrowly** | GraphState combines recent edge/wedge/geometry evidence with a strict paired seed gate, and current directed/LapPE screens isolate remaining cheap channels |
| Latest frontier mechanism | **No** | The frontier has moved to learned geometry, multi-stage supervision, larger atom/bond models, and electronic teachers |
| Safest next action today | **Finish the active screens** | K3b, directed bond memory, and the LapPE cache are already authorized; starting another tiny topology screen would not address the dominant frontier gap |

This distinction prevents a category error. The current project is not stale
because it failed to copy a 215M-parameter, 34-A100-day model. It is solving a
different constrained problem. But it must not describe the bounded result as
world SOTA, and it should not spend months adding small topology variants while
the score gap is dominated by training contract and scale.

## Anti-waste gate for every future paper-inspired route

Before a paper can create a new remote experiment, record a one-page candidate
card with all of the following fields:

1. **Target identity:** exact property, units, DFT level, and whether it is Gap,
   HOMO, LUMO, orbital, force, or energy.
2. **Role and split:** official validation, official test-dev, paper validation,
   re-segmented subset, or MolGap 100K/10K; never use a bare “PCQM score.”
3. **Inference input:** categorical 2D graph, RDKit/ETKDG conformer, predicted
   geometry, DFT geometry, or an ensemble; state whether the coordinates are
   explicit at test time.
4. **Training information:** DFT coordinates, auxiliary labels, external
   pretraining, pseudo-labels, and molecule overlap checks.
5. **Scale and cost:** parameters, number of GPUs, GPU type, wall time, number
   of stages, ensemble size, and whether code/checkpoints exist.
6. **Matched baseline:** the paper's fresh comparator and the MolGap comparator
   under the same role, seed, data and budget.
7. **Expected decision:** the one material-gain threshold and the exact reason
   to close the route.

Reject the route from the active candidate pool if any of fields 1--5 is
unknown, if its score is only from another target/split, or if the only path to
the number requires mixing a different conformer method into MolGap. That
paper may remain in the literature reserve, but it is not an experiment
authorization.

## Recommended execution policy

1. Finish K3b, the directed-bond screen, and the SignNet-LapPE cache/screen
   already authorized by `CURRENT_STATE.md` and `ROADMAP.md`.
2. Run the frozen GraphState A100 timing/memory gate. Do not interpret the
   100K/10K value as a leaderboard submission.
3. If the immediate goal remains a strict low-cost MolGap model, stop adding
   random-init micro-variants after the current authorized screens unless a
   paired candidate clears the existing `0.001 eV` material-gain rule and the
   resource gate.
4. If the actual goal is leaderboard parity, create a separate, explicitly
   authorized frontier contract before implementation. It must state which
   hard constraints are being relaxed, and it must remain separate from the
   bounded discovery result. TGT is the first reproducibility target because
   its paper and official submission are documented; DeMol and TetraGT are
   reference results until code and exact data lineage are available.
5. Any future electronic-teacher track should start with near-target
   QCDGE/HEDMoL-style labels or charges/bond order, after molecule-overlap and
   level-of-theory audits. qcMol and OMol25 are teacher/data candidates, not
   direct replacements for the PCQM Gap label.

The decision is therefore not “search forever for one more architecture.” It is
to choose explicitly between a constrained, reproducible small model and a
separate leaderboard-parity program whose geometry, pretraining, scale and
compute budget are honestly declared before training begins.
