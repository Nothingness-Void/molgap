# Protocol: V5 route portfolio

## Resource boundary

- Kaggle2 and Kaggle3 share a maximum budget of 60 GPU-hours for the week.
- Kaggle1 is reserved for desktop and must not be queried, submitted to,
  monitored, cancelled, or modified by this portfolio.
- Official validation, test-dev, test-challenge, shadow and desktop-owned jobs
  remain untouched.

## Stage 0 — zero-training reference reconstruction

Kaggle account identity is not a scientific platform difference. Kaggle2 and
Kaggle3 use the same T4 runtime family, so this portfolio must not spend two
full runs merely to repeat the same deterministic contract.

The retained K1-v4 record already contains the model, checkpoint, aligned
development prediction payload, trace and accepted runtime certificate. Stage
0 reconstructs, without training or inference, the missing reusable V5
reference bundle, portable target-transform asset, row/target manifests, role
events, acceptance, decision and RML links from those immutable artifacts.
Every reconstructed field must be directly evidenced; historical unknowns
remain unknown.

If any indispensable reference artifact cannot be verified, candidate release
stops. At most one one-time K1-v4 reference run may then be proposed under a
separate compute decision. A second account repeat is not authorized. Training
stochasticity is not inferred from account identity and no run-stability claim
is made without genuine independent repeats.

## Stage 1 — two independent architecture questions

After Stage 0 acceptance, Route A and Route B may use Kaggle2/Kaggle3. When the
two candidates are ready together, they should run as isolated workers in one
T4x2 notebook; the second account remains available for an evidence-justified
successor. Each route is a separate `architecture_comparison` with one declared
mechanism, one visible accelerator, independent RNG/optimizer/checkpoint state,
seed 42, the fixed PCQM-100K V5 contract, and the immutable Stage-0 reference.
The baseline is not retrained in candidate jobs.

Every route must freeze source/config first and then create, before submission:

1. a prospective `trajectory.json` with hypothesis, alternatives, related
   closed families, cheapest falsifier and expected native cost;
2. `training_contract.json`, role snapshot, budget snapshot and immutable cache
   acceptance;
3. a validated reference bundle and `comparison_readiness_prelaunch.json`;
4. runtime/preflight evidence including parameter count, memory and one
   optimizer-inclusive step;
5. atomic checkpoint, resume, trace, role-event and cost-event outputs.

Terminal collection must add aligned prediction/row/target manifests,
checkpoint/source/runtime hashes, observed `comparison_readiness.json`, paired
analysis, acceptance, decision and the trajectory result before RML rebuild.

## Stage 2 — attribution before another route

A negative result closes its exact mechanism and receives subgroup/trajectory
attribution before another submission. A positive result must exceed the
prospectively frozen V5 materiality rule and have a favorable paired interval.
Because training stochasticity is not measured by account duplication, one
seed can earn only a shortlist decision; it does not establish run stability
or automatically authorize another seed, 500K, full training or a protected
role.

Unused GPU-hours remain unspent until Stage-1 evidence identifies one distinct
next bottleneck. No third architecture is pre-authorized merely to consume the
weekly quota.
