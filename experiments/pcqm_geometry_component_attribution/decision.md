# Geometry component attribution diagnostic, 2026-09-28

## Accepted observation

`NO_TRAIN`. The prospective, hash-gated local diagnostic read only the two
already accepted 50,000-row development prediction payloads. Both payload
hashes, ordered source indices `500000..549999`, targets, finite predictions,
and the original acceptance MAEs were verified. The recomputed 2D MAE is
`0.1176588802 eV`; the combined distance-plus-angle MAE is `0.1141408510 eV`.
The paired gain is `0.0035180292 eV`, within floating-point tolerance of the
accepted original. The combined arm improves 52.862% of rows. No training,
checkpoint loading, model inference, remote access, or protected-role read
occurred in this action.

The target-gap strata are descriptive on the previously selected development
role. Candidate-minus-baseline MAE is `+25.624 meV` for 2–4 eV (1,877 rows),
`-4.825 meV` for 4–6 eV (32,057), `-6.479 meV` for 6–8 eV (14,712), and
`+19.008 meV` above 8 eV (1,339). Only 15 rows are below 2 eV; that bin is
too small to interpret. The gain is concentrated in the common 4–8 eV range.
Stratifying on baseline error shows larger apparent gains in its high-error
quartile, but conditioning on that error creates regression-to-the-mean bias;
it is not evidence that a geometry component causally fixes hard molecules.
The exact counts and correlations are in `analysis.json`.

## Replay and component decision

The original paired endpoint remains accepted under its frozen legacy
contract. The retained 60-epoch canonical traces have zero observed optimizer
steps, sample-presentation counts, checkpoint identities, and device times in
both arms. The RML reference index classifies the pair as
`paired_endpoint_under_frozen_legacy_contract` and `not_v5_qualified`. These
fields cannot be reconstructed from epoch numbers for a replay claim.

The model factory already supports distance-only and angle-only modes, but the
owning 500K trainer, preflight and acceptance are frozen to the original 2D
and combined pair. A new two-arm run would compare distance-only with
angle-only; without a strict 2D comparator it would not measure each component's
gain over 2D or its contribution to the already accepted combined result.
The local accepted metadata also lacks per-molecule geometry attributes for a
cheaper direct attribution. Spending roughly two historical 500K arm budgets
for that incomplete answer does not change the current model recommendation or
justify accelerator release. The original combined-geometry nomination remains
valid only within its own contract. No distance-only or angle-only result is
claimed.

Reopen only with a separately frozen, decision-relevant component contract:
an authenticated 2D comparator or a justified factorial design, accepted
executable geometry cache, exact role and native-cost ceiling, real full-batch
preflight, a training adapter that records observed step/presentation/resume
identity, and independent strict terminal evidence for every new arm. Reuse the
existing model modes and MolGap RML/acceptance infrastructure. Official
validation, test-dev and test-challenge stay sealed.
