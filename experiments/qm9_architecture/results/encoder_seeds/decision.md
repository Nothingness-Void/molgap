# QM9 encoder-seed repeats

Date: 2026-07-27

Metrics: `encoder_seeds.json`. Protocol: `../README.md`.

## Question

Every uncertainty band in the screen came from frozen-head seeds. Encoder seeds
were never repeated, yet two decisions rested on differences smaller than that
unmeasured noise:

- GPS9-192 (0.08634) versus GPS11-160 (0.08642) standalone differ by 0.00008 eV.
- As a second expert after GPS9, GPS11 beat GPS7 by 0.00115 eV against a
  head-seed spread of 0.0004-0.0005 eV.

## Setup

GPS7, GPS9-192, and GPS11-160 trained at seeds 43 and 44 under the exact
seed-42 protocol: topology geometry, 100000/10000/10000 split, split seed 42,
30 epochs, unchanged optimiser defaults. Seed 42 is the existing run.

## Result

Test average MAE across three encoder seeds:

| encoder | seed 42 | seed 43 | seed 44 | mean | std | range |
|---|---:|---:|---:|---:|---:|---:|
| GPS7 | 0.09160 | 0.09218 | 0.09349 | 0.09243 | 0.00097 | 0.00189 |
| GPS9-192 | 0.08634 | 0.08560 | 0.08705 | 0.08633 | 0.00072 | 0.00144 |
| GPS11-160 | 0.08642 | 0.08592 | 0.08609 | 0.08614 | **0.00026** | 0.00050 |

## Conclusion

**Encoder-seed noise reaches 0.00097 eV, roughly twice the head-seed spread
previously reported.** Consequences:

1. The 0.00115 eV second-expert margin is the same order as this noise, so it
   does not by itself separate GPS11 from GPS7.
2. GPS9 and GPS11-160 are indistinguishable on accuracy: three-seed means differ
   by 0.00019 eV. The earlier 0.00008 eV ordering was sampling accident.
3. The one real signal is **variance**. GPS11-160's seed spread is a third of
   GPS9's and a quarter of GPS7's.

GPS11-160 therefore remains the right fixed identity path, but for
reproducibility rather than accuracy. That is the stronger justification.

GPS7 is the only encoder genuinely separated: 0.00609 eV worse than GPS11-160,
far outside noise. Its elimination stands.

## Standing constraint

**Architecture differences below 0.001 eV average MAE cannot be treated as real
without multi-encoder-seed evidence.** Head-seed bands understate the noise.
This applies to every comparison in `../README.md` reported with head-seed
uncertainty only.
