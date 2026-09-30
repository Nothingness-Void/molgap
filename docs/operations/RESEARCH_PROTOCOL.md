# Research Protocol

Read this file when proposing or modifying a research question, accepting a
terminal result, writing attribution, making a comparison/promotion claim, or
calibrating screening/early stopping. It preserves the detailed rules moved
from AGENTS.md; frozen experiment contracts and V5 remain authoritative.
Repository-relative paths below are rooted at the selected checkout.

## New research question protocol

Before proposing, modifying, or submitting an experiment for a new scientific
question, query RML first.

Use RML to answer:

1. Has this question or mechanism already been tested?
2. Has the same or a related model family already failed under a comparable
   contract?
3. Which strict references are available?
4. Which development, validation, shadow, test, or protected roles have already
   been used?
5. What native cost evidence exists?
6. Is there already a closed, negative, inconclusive, or duplicate trajectory
   that answers the question?
7. Which evidence gaps remain genuinely unresolved?

Then read the authoritative records linked by RML.

A new experiment is justified only after that evidence review.

For a genuinely new research route, create a canonical:

```text
record_mode = prospective
```

`trajectory.json` **before** running a new diagnostic or training experiment.

The prospective record should capture the decision-relevant state required by
the active contract, including as applicable:

- research question;
- observed baseline deficiency;
- supporting evidence;
- alternative explanation;
- changed mechanism;
- cheapest decision-relevant falsifier;
- related closed families;
- reference identity;
- role-use state;
- budget/cost expectation;
- the decision the experiment is intended to change.

A trajectory may validly terminate without training.

Valid outcomes include, when supported by the active contract:

```text
NO_TRAIN
NEGATIVE_UNDER_CONTRACT
INCONCLUSIVE
STOP_FOR_COST
DUPLICATE_EVIDENCE
```

Do not launch training merely because a trajectory exists.

Advance through diagnostic, 100K, 500K, full-scale, or other gates only when
required and authorized by the active scientific contract.

After each accepted canonical research milestone, update the relevant canonical
records and rebuild RML.

Examples of milestones:

- diagnostic decision;
- terminal 100K acceptance;
- terminal 500K acceptance;
- final scientific decision;
- new native-cost record;
- new exact role-use record;
- accepted reusable trace;
- READY_FOR_DESKTOP generation.

Do not rebuild RML merely for transient runtime progress such as every epoch or
heartbeat.

Before handoff or commit, verify that RML derived state is current.

Historical evidence may be imported as:

```text
record_mode = retrospective_partial
```

only when directly supported by pre-existing authoritative records.

Never invent retrospective hypotheses, rationale, costs, role use, traces, or
evidence to justify work that was already launched. Unknown fields remain
unknown.

## Terminal attribution before another module

After accepting a module experiment's terminal result, write a concise
failure-mode attribution beside its owning decision before proposing another
unrelated module. Use the accepted contract, artifacts, trace, role and cost
records, and existing analysis helpers. This is an interpretation of evidence,
not a replacement RML schema or a new promotion gate. An infrastructure-only
failure or NO_TRAIN closure needs only the reason and missing discriminator.

For a trained module, distinguish the following where the retained evidence
permits it:

- contract, source, data, runtime, artifact, and reference validity;
- absolute and paired endpoint effect under the frozen gate;
- train and development behavior with their exact metric/EMA semantics;
- optimizer-step and sample-presentation exposure, selected endpoint, and
  whether absolute and relative gains are still moving;
- paired row errors, calibration shifts, and predeclared slices when available;
  label any additional slice or same-role fit as post-hoc and non-promotional;
- observed native cost and exact role use;
- which explanations are contradicted, supported, merely compatible, or
  unidentifiable, followed by the cheapest decision-relevant falsifier.

Do not diagnose underfitting, overfitting, insufficient exposure, or module
harm from one endpoint score or from mismatched train/development metrics.
State `insufficient_evidence` when the needed comparator, fixed-cohort metric,
trace, or checkpoint is absent. Preserve the accepted terminal decision; do
not manufacture a reference or launch a repeat solely to complete an
attribution report. Query this disposition along with RML before selecting a
new module in the same family.

## Comparison and promotion discipline

A new scientific promotion claim requires the evidence required by its active
contract, which may include:

- approved source/config identity;
- data identity;
- row/split identity;
- feature identity;
- target/transform identity;
- seed;
- precision;
- optimizer;
- scheduler;
- loss;
- exposure;
- selection rule;
- strict reference bundle;
- runtime qualification;
- finite predictions/targets;
- shape checks;
- source-index alignment;
- artifact hashes;
- resume/step cursor;
- paired comparison;
- bootstrap or other required uncertainty analysis;
- role-use history;
- actual native cost.

Missing strict reference evidence means pending, not automatic baseline
retraining.

Row bootstrap does not measure training stochasticity.

A material promotion threshold is not automatically a measured variance
estimate.

Mechanical acceptance, scientific interpretation, transfer qualification,
budget decision, and full-scale handoff are separate states.

Do not collapse them into one `accepted=true`.

## READY_FOR_DESKTOP

`READY_FOR_DESKTOP` is a strict evidence package, not a generic positive label.

Generation is fail-closed under the active RML/V5 contract.

A server-side candidate does not become READY merely because a 100K or 500K
metric is positive.

The package must satisfy the required prospective trajectory, qualification,
reference, role, cost, artifact, comparison, and provenance requirements.

A READY package:

- does not wake the desktop;
- does not submit full-scale work;
- does not modify production;
- does not consume protected roles by itself.

Desktop independently decides what to do with READY evidence when it is online.

## Training-trace and early-stop discipline

Do not enable a new early-stop or screening policy merely because partial
training curves exist.

RML backtesting requires strictly comparable trace identities under the active
analysis contract.

Relevant comparability may include:

- scientific contract;
- dataset identity;
- split/row identity;
- architecture identity;
- optimizer;
- LR schedule;
- precision;
- EMA semantics;
- target transform;
- evaluation/selection role;
- terminal endpoint;
- x-axis semantics.

Epoch numbers alone are not generally comparable across different dataset
sizes or training contracts.

Prefer decision-relevant coordinates such as:

- optimizer steps;
- sample presentations;
- matched frozen prefixes.

If the available historical traces are insufficient, the correct result is:

```text
insufficient_evidence
```

Do not replace missing rates or cost savings with zero.

A future screening or early-stop policy must be calibrated prospectively before
activation.

## Default scientific workflow

For a new research question, the default reasoning path is:

```text
define observed deficiency / question
    ->
query RML for prior evidence and related closed families
    ->
follow RML pointers to authoritative records
    ->
decide whether existing evidence already answers the question
    ->
NO_TRAIN if no new experiment is decision-relevant
    ->
otherwise create prospective trajectory
    ->
define hypothesis and cheapest falsifier
    ->
run the cheapest justified diagnostic
    ->
if justified, release the next bounded training action
    ->
strict acceptance
    ->
scientific interpretation
    ->
cost / role / trace recording
    ->
RML rebuild
    ->
decide zero or one next justified action
```

For the current V5 server screening funnel, when applicable:

```text
research question
    ->
evidence retrieval
    ->
prospective hypothesis card
    ->
cheap diagnostic
    ->
100K
    ->
strict acceptance + decision
    ->
500K only if qualified and authorized
    ->
strict transfer/cost decision
    ->
READY_FOR_DESKTOP only if fully qualified
    ->
stop server scale-up
```

Desktop full-scale, official evaluation, final submission, and production
promotion remain separate decisions under their owning contracts.

The objective is not to keep accelerators busy.

The objective is to obtain the minimum new evidence required to make the next
scientifically justified decision.

