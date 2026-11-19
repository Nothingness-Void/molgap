# DeMol-CIN 100K decision

2026-11-19 JST. Run pcqm-demol-cin-100k-20261118-r2-s42, attempt v1.
Candidate version2 continues the original checkpoint from 42636 to 46860 updates;
reference pair_update_norm keeps its complete version1 result. Both arms finish
60 passes, 46860 updates and 5998080 sample presentations. Source, Spec, data,
recipes and main-EMA selection stay frozen. The recovery repair changes
validation and transport only.

## Outcome

- pair_update_norm: CLOSED (same-job reference).
- demol_cin: INCONCLUSIVE. Strict comparison is not ready, so this is not a
  causal positive and not a negative architecture result.

## Numbers

The frozen primary is selected EMA 0.999 on the same 50000 internal development
rows. Reference MAE 0.139440192079 eV (pass42); candidate 0.134635221485 eV
(pass35). Candidate minus reference is -4.804970593 meV, paired row-bootstrap
95% CI [-5.939758524, -3.724133322] meV (2000 samples, seed 42). Both frozen
conditions pass: point delta <= -3 meV and CI upper < 0.

This is a numerical result, not strict evidence. Do not cite it as causal
evidence for the architecture.

## Why strict is blocked

The comparison stops on SCIENTIFIC_CONTRACT_MISMATCH in runtime_certificate_scope:
reference c75011519274a52cd40b1610d8b967276fc8fcad5887ab4ca2b68969533ba539,
candidate f0b18183bd71f2cc42035a02dffd66bbea33b6acf0fa78f204fa59b47eb254f4.
Both arms ran in one kernel session (kernel 145779299, version 1) with the same
source, data, recipe, seed and GPU type (T4). The candidate keeps the same hash
in v1 and v2, so continuation did not cause the gap.

The cause is not proven. It most likely comes from per-card properties; this is
an inference. The retained records hold only hashes, not the underlying lists.

## Limits

- The numeric effect of the runtime difference was not measured.
- The bootstrap resamples rows only. It does not include training randomness;
  100K init spread is about 1.6 meV.
- The auxiliary EMA 0.9999 is descriptive. It does not replace the frozen rule.

## Cost

The candidate's final pass (MAE 0.138426214457 eV) leads the reference final
pass (0.139866888523 eV) by only 1.440674067 meV. Candidate training wall is
17155.307590 s against 9857.681168 s, 1.7403 times the reference. Physical
allocation is 10.146649125 T4-device hours (v1 16028.011608 s plus v2
2235.956818 s), measured from Python entry; queue time is unmeasured.

## Release

This result does not enter the 500K, full-data, production or replay pools. No
successor training is authorized here. Official validation, test-dev and
test-challenge were not used (see the role audits in closure_v2).
