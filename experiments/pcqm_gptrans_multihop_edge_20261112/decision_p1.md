# P1 paired-noise calibration decision, 2026-11-14

## Scope
Covers runs s50-s54 (attempt v1, five E1-identical pairs ref_sNN + mh_sNN; init seed is the only change from E1). Protocol Amendment 4 defines P1 as a noise calibration, not a candidate test: "No P1 outcome changes the E1 verdict or reopens multi-hop at 100K; only E2/W2 decide that line." This file does not touch the E1 ruling in decision_e1_n1.md, the E2 gate, or any frozen threshold. E2 (s47/s48) is not covered.

Closure outcomes: every ref_sNN is CLOSED; every mh_sNN is INCONCLUSIVE with failure_mode null (calibration pair, no gate applied). The per-pair effect_meV in closure facts is the measured paired delta, recorded descriptively.

## Acceptance
All five runs: strict read-only recheck MECHANICALLY_VERIFIED for both arms; only the human-field blockers (outcome/decision_ref, protected-role audit) remained. draft-closure showed development, official_validation, test_dev and test_challenge roles `untouched` in every arm (no data-open evidence beyond development shards). No run was dropped. Role used: `fixed100k-next50k` (50,000 rows) only; tail_holdout_v1, official validation and test were not read.

## Evidence
Development MAE in meV, best-development-EMA endpoint. d = mh - ref (negative is better). s43/s44 from E1 closures, s45/s46 from N1 closure.

| Seed | ref dev MAE | mh dev MAE | d (mh - ref) |
|---|---|---|---|
| s43 | 138.51349 | 139.17193 | +0.65845 |
| s44 | 139.28710 | 138.62045 | -0.66666 |
| s45 | 140.00867 | | |
| s46 | 142.21825 | | |
| s50 | 140.19321 | 139.16894 | -1.02427 |
| s51 | 138.51346 | 139.54902 | +1.03556 |
| s52 | 138.15969 | 139.53080 | +1.37110 |
| s53 | 138.16764 | 139.33671 | +1.16907 |
| s54 | 139.32264 | 138.93866 | -0.38398 |

## Statistics (frozen in Amendment 4)
- S_ref (sample SD, ddof=1, nine refs) = 1.3031 meV (range of the nine = 4.059 meV)
- S_pair (sample SD, ddof=1, seven deltas) = 0.9770 meV
- m7 (mean of seven deltas) = +0.3085 meV (descriptive)
- sqrt(2) * S_ref = 1.8429 meV; 0.5 * sqrt(2) * S_ref = 0.9214 meV; 0.8 * sqrt(2) * S_ref = 1.4743 meV
- S_pair / (sqrt(2) * S_ref) = 0.530

## Frozen reading
- S_pair = 0.9770 is above 0.9214 (not <= 0.5 * sqrt(2) * S_ref) and below 1.4743 (not >= 0.8 * sqrt(2) * S_ref): UNDETERMINED. Report both sizing bases: shared-init pairing cancels somewhat under half of the init noise by this estimate (ratio 0.53, just outside the "at least half" band); a screen sized from S_pair would need 2 * S_pair / sqrt(n) <= gate, one sized from S_ref needs 2 * sqrt(2) * S_ref / sqrt(n) <= gate.
- S_ref = 1.3031 meV is above 1.3 and below 2.0: neither frozen reading fires. N1's D = 3.705 meV is not declared a tail draw (S_ref marginally above 1.3, by 0.003 meV), and the "two-pair screens with gate below 2 * S_ref are underpowered" reading does not fire either.
- m7 = +0.31 meV is descriptive only.

## Effect on other decisions
The E1 verdict (NEGATIVE_UNDER_CONTRACT for mh_s43, mh_s44, with the N1 confidence note) is unchanged. Multi-hop at 100K is not reopened or closed by P1; only E2/W2 decide that line. The row-bootstrap CIs in the closure facts are not seed variance and are not used here.
