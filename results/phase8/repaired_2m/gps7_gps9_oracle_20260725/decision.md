# Repaired-2M GPS7/GPS9 Oracle

## Decision

The GPS7/GPS9 molecular hard-expert path passes its Oracle-only feasibility
gate. This authorizes scaffold-disjoint OOF gain-label generation, not Router
training or deployment.

## Target-Specific Switch Oracle

Average-MAE changes are relative to the GPS7 base. Negative values improve.

| Scope | 5% calls | 10% calls | 20% calls | Unconstrained |
|---|---:|---:|---:|---:|
| Common | -0.003409 | -0.005355 | -0.008126 | -0.013654 |
| OOD-1000 | -0.003683 | -0.005671 | -0.008451 | -0.013816 |
| P8-hard | -0.003114 | -0.005027 | -0.007794 | -0.013489 |

At the required 10% call budget, P8-hard improves by `0.005027 eV`; common and
OOD also improve. Expected cost is `1.10` encoder passes per molecule when
GPS7 is always run and GPS9 is called for 10%.

These are perfect-label ceilings. The external common/OOD/P8-hard labels are
evaluation-only and are forbidden as Router training labels. The next valid
step is to generate GPS7 and GPS9 predictions out of fold on repaired-2M
scaffold-disjoint training folds.

Metrics: `oracle_metrics.json`.

No sealed-20K rows were opened and the production registry is unchanged.
