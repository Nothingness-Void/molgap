# Expert specialization reading, 2026-09-30

Scope: molecular specialization/routing, including a January 2026 preprint.
Four full-text methods/experimental sections were read; two additional recent
works are abstract-only leads because full text was unavailable. This is a
targeted review, not a claim to cover every frontier paper or leaderboard.
The applicability judgments below are MolGap interpretations.

## 1. TopExpert (AAAI 2023)

[Full text](https://arxiv.org/html/2302.13693), methods equations 3-11 and
experimental tables 1-2. A shared GNN feeds small classification experts.
Student-t cluster assignments, annealed Gumbel-Softmax, KL cluster cohesion,
and scaffold alignment determine specialization; inference uses deterministic
soft assignments. Eight MoleculeNet classification datasets support topology
grouping, not PCQM Gap regression. GCN average AUC is 70.5 versus 69.7 for
equal expert averaging. Effects vary by backbone/dataset.

Transfer idea: share a pretrained encoder, retain global context, and let
small heads emphasize regions. Gaps: regression objective, stable regions,
minimum expert exposure, and unseen-scaffold behavior require qualification.
Scaffold membership does not demonstrate predictable K1/GPTrans winner identity.
Cheapest falsifier: inspect local Oracle headroom and held-out routing regret
before training any topology experts. Prior coarse-feature AUC 0.534 is an
adverse local comparator, not a universal impossibility result.

## 2. GNN-MoCE (2023 preprint)

[Full text](https://arxiv.org/html/2312.03292), sections III-V. Expert-specific
SAG pooling gives different graph readouts; attention diversity and individual
expert losses address homogenization and dominant experts. Noisy top-k routing
also uses task-description embeddings. Experiments jointly train 35
classification tasks from 24 datasets, reporting mean AUC 0.809; baselines
have different single-task or pretraining conditions. This is not a controlled
single-Gap gain estimate. The printed split description is inconsistent
(develop 80%, test 80%); implementation must be checked before reproduction.

Transfer idea: different readouts and explicit expert gradients may preserve
complementarity after pretraining. Single-task PCQM has no varying task
description to route on. Diversity penalties cannot substitute for accuracy,
and balanced utilization is not evidence of distinct chemical competence.
Cheapest falsifier: use existing prediction-space routing; only later compare
diverse heads against equal-parameter ordinary heads under the same exposure.

## 3. ASE-Mol (2025 preprint)

[Full text](https://arxiv.org/html/2504.05844), equations 4-17, Algorithm 1,
tables 1-3. BRICS masks generate fragment representations; label-dependent
attribution ranks positive/negative motifs. Recognition training precedes
motif-specific experts, with triplet and importance losses. Eight datasets
are classification-only, with ten seeds and scaffold splits. Removing motif
learning strongly reduces reported SIDER/HIV AUC. Some baseline numbers are
inherited from earlier work; large gains require independent replication.

The attribution formula branches on Y=0/1. Directly copying it to continuous
Gap or selecting test motifs using true labels would be invalid. A new
training-only regression attribution contract would be necessary. Masked
representations are not causal interventions on molecular chemistry.
Transfer idea: specialists may attend to fragments while retaining full context.
Cheapest falsifier: evaluate retained embedding/fragment discrimination only
after justified inputs exist; do not introduce BRICS expert training now.

## 4. MI-MoE (January 2026 preprint)

[Full text](https://arxiv.org/html/2601.12637), sections 3-4, Algorithm 1,
tables 4-6 and Appendix B. Experts use different distance cutoffs; filtration
descriptors and Betti curves feed a top-k gate. ETKDG/MMFF conformers support
eight MoleculeNet and polymer tasks; no PCQM Gap experiment is shown.
Backbone depth changes between standalone and expert configurations.
One-expert SchNet also improves in their ablation, so the complete gain cannot
be assigned solely to cooperation. The forward pseudocode computes every
expert despite sparse weights; sparse dispatch speedup needs implementation
and measured timing. Published averages mix different regression properties.

Transfer idea: specialize interaction ranges with inference-available geometry.
Local geometry-blend incremental gain remains uncertain, and additional
conformer/topology preprocessing adds cost. Preserve MolGap geometry identity;
do not copy their curation/neutralization onto PCQM. Cheapest falsifier: retained
geometry versus 2D predictions and conditional Oracle budgets; no new cutoff
experts until realizable gain and native cost are qualified.

## 5. MoL-MoE (abstract-only lead)

[Author source](https://research.ibm.com/publications/multi-view-mixture-of-experts-for-predicting-molecular-properties-using-smiles-selfies-and-graph-based-representations--1)
describes pretrained SMILES, SELFIES and graph representations. OpenReview PDF
access returned a browser-verification page. The full experimental controls,
pretraining overlap, inference dispatch and code were not verified here.
Do not treat architectural views as independent information by definition.
Potential fit is heterogeneous pretrained fusion; implementation is deferred.

## 6. TAME (September 2026 preprint; abstract-only lead)

[Author/institution record](https://www.hsbi.de/publikationsserver/record/7152)
describes graph/text/descriptors, coordinate-wise gates, balance/entropy
regularization and graph pretraining. It reports BACE scaffold experiments
with 100 seeds and reduced collapse tails under matched fusion controls.
The available record is an abstract, not verified full methodology. Evidence
concerns small-data classification reliability, not PCQM MAE. Coordinate-wise
representation fusion requires retained compatible embeddings; our prediction
Oracle does not test it. Do not claim a replicated mechanism or expected gain.

## Synthesis and experiment choice

Separate three claims: errors differ; winner/weight is learnable without labels;
training specialists creates useful differences. Only the first two are tested
by retained predictions. Shared-encoder head specialization is cheaper than
several full encoders, but cannot be evaluated from scalar predictions alone.
Use the local protocol to determine headroom and realizability. Then await
pretraining, test its fixed fusion, and consider at most one independently
qualified specialization question. No paper releases a remote job.
