# V4 comparability audit

Audit date: 2026-09-12

## Scope

This record compares the submitted geometry-transfer 500K chain with the
reference-screen V4 authority in `../SCREENING_POLICY.md`. It changes neither
the frozen experiment nor the running SCNet jobs.

## Asset inventory

The repository has four accepted scientific dataset identities derived only
from the official PCQM4Mv2 training role: 100K, matched 500K/50K, 1M, and full
training. IMS retains all four. Kaggle1 and Kaggle2 each retain byte-identical
accepted 100K and 500K graph mirrors. These are V4-compatible input assets,
not V4 model results.

The submitted chain uses the accepted matched-500K geometry aggregate
`676a506c808402bc16a4437cc02286168239a8dd5de4451e992131eddb4f1b20`.
Its source ranges, OGB atom/bond categories, RWSE16, ETKDGv3+MMFF94s geometry,
labels, and sealed-role flags match the fixed identity.

## Contract audit

| Requirement | Submitted chain | Status |
|---|---|---|
| Accepted fixed data and payload hash | exact matched-500K aggregate | pass |
| Physical batch per device | 128 | pass |
| One device and no accumulation | yes | pass |
| Tail policy | per-shard `drop_last`; 3,900 steps/epoch | pass |
| Seed and precision | seed 42, FP32 | pass |
| Sealed official roles | validation/test roles unread | pass |
| Atomic recovery | model, optimizer, schedule, RNG, trace | pass |
| V4 runtime certificate | not emitted | fail |
| Deterministic algorithms and repeated-step calibration | not enforced | fail |
| TF32/runtime/software fingerprints | not recorded as a V4 certificate | fail |
| Frozen matching V4 reference | none for either architecture contract | fail |
| Same optimizer/schedule across the two candidates | no | fail for direct comparison |

The successful preflight files are memory and finite-backward gates. They are
not `molgap-runtime-certificate-v1` certificates and cannot be relabeled as
such. The training source does not enable the V4 deterministic-algorithm guard,
so a later calibration cannot retroactively promote these runs.

## Evidence classification

No completed model result in this line is currently an eligible immutable V4
cross-platform reference. The historical K1 500K pair remains paired-V3
evidence because every epoch included a 32-row tail batch. The historical
GPTrans-T and ESGPS6-304 comparison also predates the V4 reference certificate
and no-tail acceptance.

When complete, the two submitted geometry runs remain useful same-data
nomination evidence. Their absolute development scores and fusion headroom may
select one route for confirmation. They cannot establish causal geometry gain
against historical pure-2D scores, and their mutual difference combines
architecture with optimizer, schedule, epoch exposure, and EMA differences.
The development-trained scalar fusion is selection evidence, not an unbiased
final estimate.

## Economical V4 bridge

Do not rerun both geometry routes automatically. First finish the submitted
chain and nominate at most one route. For that route only:

1. Freeze one pure-core reference and one geometry candidate with identical V4
   scientific fields except architecture.
2. Generate and accept one reusable runtime certificate for every actual
   platform/software/runtime tuple before training.
3. Train the missing pure reference once and rerun the nominated candidate
   under deterministic V4 execution; never retrain the reference per platform.
4. Apply the stochasticity floor and material-gain gate before opening any
   disjoint audit role.

With those controls, V4 explicitly permits the reference and candidate to run
on different accelerators or platforms. Without them, matching dataset hashes
alone permits contextual comparison only.
