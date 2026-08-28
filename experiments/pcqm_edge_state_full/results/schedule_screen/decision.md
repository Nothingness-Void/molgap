# Schedule Screen Decision

Decision date: 2026-08-27

The official-train-only schedule screen accepted the twenty-epoch schedule
with two warmup epochs followed by cosine decay. It reused the same accepted
OGB categorical graphs, `100K` train rows, `10K` development rows, and seeds
42/43/44 as the rich-feature screen. Official validation and test were not
read.

| Schedule | Seed 42 | Seed 43 | Seed 44 | Mean development MAE |
|---|---:|---:|---:|---:|
| 10-epoch cosine | 0.174358 | 0.180146 | 0.177425 | 0.177310 eV |
| 20-epoch warmup/cosine | 0.150062 | 0.150637 | 0.151366 | 0.150688 eV |

The paired improvement was `0.026621 eV` overall, `0.048497 eV` on radicals,
and `0.024812 eV` on non-radicals. All three seeds improved; the best epochs
were 16, 19, and 18. The declared mean, direction, and maximum-regression gates
all passed.

This result selected the bounded twenty-epoch schedule for a possible full
rich-feature attempt. It did not establish an official PCQM score. The earlier
forty-epoch feature runs reached lower internal MAE, so the selected schedule
is a compute-bound compromise rather than the unconstrained internal optimum.

The retrieved machine acceptance is
`platforms/_records/ims/pcqm_schedule_screen_20260827/schedule_acceptance.json`,
SHA256 `6c987ed83757dde17aa8e7682b58536ff246cd7fa9792d5bc582f17263508e60`.
