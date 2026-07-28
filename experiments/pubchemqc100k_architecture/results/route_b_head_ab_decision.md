# Bounded-Fusion Head A/B

## Decision

Replace the shared gated-sum bottleneck with a bounded residual head for the
full-scale precision architecture.

The retained head preserves GPS11-160 as a direct prediction path and adds a
small multi-expert correction. A subsequent validation-only scale A/B selected
`+-0.10 eV` over `+-0.25 eV` and `+-0.50 eV`; see
`route_b_residual_scale_decision.md`. The four encoder inputs and strict
two-SchNet-pass inference contract are unchanged.

## Three-Seed Evidence

| Head | Validation average | Test average | Test Gap |
|---|---:|---:|---:|
| Gated sum | 0.138117 | 0.138046 | 0.165819 |
| Concat | 0.137842 | 0.137678 | 0.165657 |
| Bounded residual | **0.135657** | **0.135901** | **0.162025** |

Bounded residual improves validation average, test average, and test Gap by
`0.002460/0.002144/0.003794 eV` relative to the gated head. All three fusion
seeds improve test average.

This head result addresses the previously observed failure mode where correlated
expert embeddings were compressed through one shared bottleneck. It authorizes
the bounded residual family for the repaired-2M scale-up protocol; the frozen
correction scale and final metrics are recorded in
`route_b_residual_scale_decision.md`.

Machine-readable evidence: `route_b_head_ab_summary.json`.

No sealed-20K rows were accessed and the production registry is unchanged.
