# Rich-Feature Screen Decision

Decision date: 2026-08-27

The official-train-only `100K` train plus `10K` development screen accepted
the complete OGB categorical atom and bond contract. Official validation and
test roles were not read.

| Schema | Seed 42 | Seed 43 | Seed 44 | Mean development MAE |
|---|---:|---:|---:|---:|
| Legacy | 0.183533 | 0.183399 | 0.182212 | 0.183048 eV |
| OGB categorical | 0.140107 | 0.141358 | 0.141830 | 0.141098 eV |

The paired mean changes were `-0.041950 eV` overall, `-0.421098 eV` on
radicals, and `-0.010587 eV` on non-radicals. Every seed passed the declared
overall, radical, and non-radical gates. This isolated missing categorical
chemistry as a material cause of the failed strict official run; it did not
authorize a production change or establish an official PCQM score.

The machine-readable acceptance is `acceptance.json`, SHA256
`329171126eebc5d130ea9c62af52e24bf736416ac36d344dab41835c694ad2a0`.
The subsequent schedule question is governed by `ROADMAP.md`.
