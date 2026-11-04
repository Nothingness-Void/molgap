# K1 init-only noise measurement — 2026-11-04

The exact kaggle3 kernel `nvoid912/molgap-k1-init-noise-server-100k-s45`,
ID142533638/version1, completed both authorized arms. Both arms passed
mechanical verification with the observed protected-role records untouched.
This record reports a measurement defined in advance in `protocol.md`; it has no
gate, no promotion and no causal claim.

## Endpoint

Both arms ran the K1 post-BN recipe unchanged except for the initial state
(seeds 45 and 46), completed the 40-pass horizon, returned 0 from preflight and
training, and wrote 40 trace observations each with `learning_rate` and a
distinct `checkpoint_identity`. The ledger records 6,719 s wall and 3.7326
allocated T4-hours, inside the 4.5 T4-hour ceiling.

## Measurement

| Initial-state seed | Selected development MAE, meV | Source |
|---:|---:|---|
| 42 | 141.2945 | reference; reproduced bit-identically on three platforms and runs |
| 43 | 139.6632 | desktop `pcqm_k1_init_noise_100k` |
| 44 | 140.6475 | desktop `pcqm_k1_init_noise_100k` |
| 45 | 141.5992 | this run, same-run reference arm |
| 46 | 142.5384 | this run |

Over the five initializations: mean 141.1486 meV, range 2.8752 meV, sample SD
(n-1) 1.0739 meV with a chi-square 95% interval of [0.643, 3.087] meV at four
degrees of freedom, paired-delta SD sqrt(2) times the SD = 1.5188 meV. The frozen
3 meV material-gain threshold is 1.98 paired SD (2.79 SD). Within this run,
`init_s46` minus `init_s45` was +0.9392 meV with a 2,000-draw paired row
bootstrap 95% interval of [+0.044, +1.805] meV; relative to seed 42 the two new
arms differ by +0.3047 and +1.2439 meV.

The K1 decay-timescale screen's difference of -0.1934 meV is 0.13 paired SD
under this estimate.

## Outcome

The seed-42-equivalent reference arm `init_s45` closes `CLOSED`. `init_s46`
closes `NEGATIVE_UNDER_CONTRACT`, the label desktop gave its seed-varied arms; it
records a finite, worse-than-reference result of a seed change, not a conclusion
about any mechanism.

## Limits and not claimed

Only the initial state varies. Data order and dropout were fixed, so this is
init-only variance and the full seed variance is likely larger. The
development cohort is reused for checkpoint selection, so absolute values are
optimistic. Five samples give only four degrees of freedom, so the SD estimate
remains wide. Seeds 42 to 44 come from records that were not produced in this
run. No claim is made that any recipe or architecture helps, and no seed beyond
45 and 46 was authorized or run.
