# Decision

Outcome: wd 0.05 (candidate) vs wd 1e-5 (reference), Muon, K1 500K, init seed 42, 60 passes (continued segment, v2).

Selected development MAE (meV): reference 101.86, candidate 100.47; delta -1.39 meV, paired bootstrap 95% CI [-2.02, -0.77] (internal_development, 50,000 rows, init-only variance, post-hoc measurement).

Reading under the frozen protocol: the per-seed gate is delta <= -3 meV with 95% upper < 0. This seed misses the gate, the sign is favourable and the upper bound is below 0. Candidate arm: POSITIVE_BELOW_GATE; reference arm: CLOSED. The joint nomination rule (both seeds) is evaluated across the s42 and s43 runs; the rule is unchanged. No single seed nominates.

Not selected on official validation or the tail holdout. Tail scoring of the final models is secondary and separate.
