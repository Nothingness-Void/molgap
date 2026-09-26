# GPTrans local inductive bias 100K screen

This desktop-owned Kaggle1 experiment asks whether a sparse, persistent
real-bond EdgeState local path improves a GPTrans-T model that already receives
RWSE16 node information. The two new arms share one T4x2 job but train and
produce evidence independently:

- **A — `rwse16`:** GPTrans-T with RWSE16 at the node input.
- **B — `rwse16_local_edge`:** A plus persistent local message passing on
  real molecular bonds. B does not add dense all-pairs attention or geometry.

The frozen comparison is B against same-job A on identical 50,000 internal
development rows. The accepted GPTrans-T 100K reference is historical context,
not a newly trained control or a strict causal comparator for A. Development
has already been reused for research selection, so a passing result can only
shortlist the mechanism for a separately contracted decision.

Read [protocol.md](protocol.md) for the question and decision gate,
[training_contract.json](training_contract.json) for the frozen numeric contract,
and [STATUS.md](STATUS.md) for release state. Canonical prospective and terminal
RML records must be generated through the existing experiment CLI after the
source and ExperimentSpec are frozen; this directory does not itself claim a
submitted job or replay-ready evidence.

Evidence basis: the accepted [GPTrans-T 100K decision](../pcqm_gptrans_t_100k_v4/decision.md)
identifies RWSE16 and a persistent sparse real-bond path as omitted mechanisms.
The matched [500K ablation decision](../pcqm_500k_v4_evidence/local_ablation_decision.md)
rejects dense global attention on the EdgeState backbone and finds a favorable
but sub-threshold molecular-slot contribution. Those results motivate this
bounded test; they do not predict its outcome. Follow the
[RML index](../../research_memory/README.md) to each canonical decision before
making a scientific claim.
