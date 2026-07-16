# Closed Phase 8 Experiment Scripts

These historical scripts support reproduction of experiments that failed their
promotion gates. They are kept out of the active Phase 8 workflow and must not
be used as production training or inference entry points. Their generated
results default to `results/phase8/archive/`.

| Directory | Result archive |
|---|---|
| `archive-r01-learned-router/` | `results/phase8/archive/archive-r01-learned-router/` |
| `archive-r02-pubchemqc-router/` | `results/phase8/archive/archive-r02-pubchemqc-router/` |
| `archive-r03-three-expert-moe/` | `results/phase8/archive/archive-r03-three-expert-moe/` |

The only active Phase 8 candidate scripts are in
`scripts/phase8/dual2d_static_candidate/`.
