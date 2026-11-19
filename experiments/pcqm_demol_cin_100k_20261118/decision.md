# DeMol-CIN 100K proposed terminal decision — closure blocked

2026-11-19 JST. Run pcqm-demol-cin-100k-20261118-r2-s42, attempt v1.
Candidate version2 continues the original checkpoint from42636 to46860 updates;
reference pair_update_norm retains its complete version1 result. Both finish
60 passes, 46860 updates and5998080 sample presentations. Original source,
Spec, data, recipes and main-EMA selection stay frozen; the attested recovery
repair changes validation/transport only.

## Proposed disposition (not finalized)

- pair_update_norm: CLOSED (same-job reference).
- demol_cin: INCONCLUSIVE under strict comparison; retain the positive numerical100K evidence.
  No500K/full training or production adoption is released by this decision.

The frozen primary is selected EMA0.999 on the same50000 internal development
rows. Reference MAE0.139440192079eV (pass42); candidate0.134635221485eV
(pass35). Candidate minus reference is-4.804970593meV, paired row-bootstrap
95% CI[-5.939758524,-3.724133322]meV (2000 samples, seed42).
Both numerical conditions pass: point delta<=-3meV and CI upper<0.
Strict comparison is blocked by SCIENTIFIC_CONTRACT_MISMATCH on
runtime_certificate_scope: reference c75011519274a52cd40b1610d8b967276fc8fcad5887ab4ca2b68969533ba539
versus candidate f0b18183bd71f2cc42035a02dffd66bbea33b6acf0fa78f204fa59b47eb254f4.
The candidate has the same fingerprint in original v1 and continued v2, so
continuation did not introduce this difference. The current retained records
do not resolve its constituent cause. Architecture is the only declared
intervention, so strict causal nomination is not granted retrospectively.
This outcome is not a negative architecture verdict.
The auxiliary EMA0.9999 is descriptive, not a substituted selection rule.

## Interpretation and costs

See attribution.md and closure_v2/diagnosis.json for the measured limits.
Candidate final-pass MAE0.138426214457eV is worse than its selected pass35;
the selected gain persists but late training does not keep improving. The
reference final-pass MAE is0.139866888523eV, so the final-pass advantage is
only1.440674067meV. This is not evidence to extend exposure blindly.

Candidate cumulative training process wall17155.307590s versus reference
9857.681168s (1.7403x). Physical allocation includes idle/qualification time
and both T4s: v1 16028.011608s plus v2 2235.956818s, totaling
10.146649125 T4-device hours. These are measured Python-entry allocation
windows; queue and provisioning before entry remain unmeasured.

Strict native/replay, exact-resume proofs, original trace prefix, actual role
use and both cost segments are checked by the existing terminal owner before
RML closure. This decision does not replace those checks. Official validation,
test-dev and test-challenge were not used according to retained data-open
provenance and the generated role audits. No protected-role score is claimed.

The strict execute attempt verified native qualification for both arms but
returned accepted_replay_evidence_missing_or_invalid for the candidate because
the runtime-mismatched comparison is not strict-ready. No terminal RML
finalization succeeded. This document records the proposed disposition and
numerical evidence, not completed acceptance.
