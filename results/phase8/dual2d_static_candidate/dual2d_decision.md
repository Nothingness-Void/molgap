# Phase 8 Dual-2D Static Candidate Decision

## Question

After removing the weak Geometry SchNet expert, do independently initialized
Local GINE6 and Global GPS9 experts form stable complementarity? The gate was
pre-registered as at least 0.001 eV Gap improvement over the best single expert
with the same direction for all three complete model seeds.

## Complete-stack three-seed result

| method | seed 42 gain | seed 43 gain | seed 44 gain | mean gain | passes |
|---|---:|---:|---:|---:|---|
| Equal average | -0.001092 | +0.012986 | +0.001715 | +0.004536 | no |
| Static weights | +0.001303 | +0.012029 | +0.003400 | +0.005577 | **yes** |
| Embedding concat Fusion | -0.003535 | +0.009765 | +0.001941 | +0.002724 | no |
| Target-specific soft gate | -0.001680 | +0.008284 | +0.004856 | +0.003820 | no |
| Static-centered soft gate | -0.001491 | +0.009778 | +0.004739 | +0.004342 | no |

Positive gain means lower internal-test Gap MAE than that seed's best single
expert. Every seed uses independently trained Local/GPS experts and its own
head/gate initialization. All rows use the same scaffold-disjoint split.

Static target-wise weights pass the requested magnitude and direction rule.
Their seed-wise Gap Local/GPS weights are approximately 0.712/0.288,
0.633/0.367, and 0.608/0.392. Seed 42's paired-bootstrap CI still crosses zero;
seeds 43 and 44 are significant. This is a promising pilot signal, not a
production claim.

## Decision

- **Dual-2D complementarity: pass at 30k.** Keep target-wise static blending as
  the only active candidate.
- **Dynamic MoE: fail.** Neither a uniform-initialized nor static-centered gate
  improves all three seeds. Do not joint-finetune a Router.
- Geometry remains removed. Do not reopen the archive-r03 three-expert route.
- Do not open sealed sets or replace routed-v4.
- Before any 700k commitment, test the three static blends on common/OOD/P8-hard
  and PCQM-like distributions. Scale only if transfer direction is stable.

Exact metrics: `results/phase8/dual2d_static_candidate/dual2d_three_seed_metrics.json`.
