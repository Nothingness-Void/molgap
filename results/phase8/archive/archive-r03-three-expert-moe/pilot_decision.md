# Phase 8 archive-r03 Heterogeneous MoE Pilot Decision

## Scope

This is a 30k architecture feasibility gate, not the proposed 700k final run.
All Local GINE, Global GPS9, and Geometry SchNet weights were randomly
initialized. No historical checkpoint was loaded. The 26,774/1,618/1,608
train/validation/internal-test split is scaffold-disjoint; the archive-r02 sealed
random and hard sets were not inferred or opened.

## Expert gate

| model | HOMO MAE | LUMO MAE | Gap MAE |
|---|---:|---:|---:|
| Local GINE | 0.164262 | 0.168298 | 0.207524 |
| Global GPS9 | 0.191820 | 0.184359 | 0.227285 |
| Geometry SchNet | 0.254195 | 0.270080 | 0.381136 |
| Equal average | 0.178290 | 0.174412 | 0.233222 |
| Validation-fit static weights | 0.164276 | 0.161864 | 0.206745 |
| Oracle | 0.110432 | 0.102076 | 0.132293 |

The Gap Oracle has 0.074452 eV headroom over static weights and all three
experts win individual molecules. This is potential complementarity, not proof
that the winner is predictable. Static weights improve Local Gap by only
0.000779 eV and the paired-bootstrap confidence interval crosses zero.

## Frozen-expert Router gate

| variant | mean Gap improvement vs static | three-seed direction |
|---|---:|---|
| shared M0, no descriptors | -0.000449 eV | all worse |
| target-specific M0, no descriptors | -0.001275 eV | all worse |
| target-specific M0 + descriptors | +0.000445 eV | mixed |
| target-specific M1 + residual | -0.000615 eV | mixed/worse |

The best M0 seed gains 0.000992 eV, but every seed's confidence interval
crosses zero and seed 44 regresses. M1 residual magnitude is about 0.056-0.062
eV and does not improve the result. Effective expert counts remain about
1.9-2.2, so this is not Router collapse.

## Decision

**STOP after the frozen-expert Router pilot.** The best mean gain is 0.000445
eV, below the pre-registered 0.0005 eV stop threshold and the 0.001 eV Router
success threshold. Do not run head-level/full-model joint fine-tuning, do not
build the 700k dataset for this architecture, and do not open sealed sets.

This rejects the current archive-r03 specification at pilot scale; it does not prove
that every future heterogeneous ensemble is impossible. Reopening requires a
new mechanism that first makes the Geometry Expert competitive or demonstrates
Router-predictable gain on a larger expert-only pilot.

## Dual-2D follow-up

The subsequent preliminary dual-2D candidate follow-up removed Geometry and retrained complete Local
GINE6 + Global GPS9 stacks for seeds 42/43/44. Target-wise static weights pass
the requested `0.001 eV` all-seed rule, but concat Fusion and both soft-gate
forms do not. This reopens only a static dual-2D candidate, not dynamic MoE.
See `results/phase8/dual2d_static_candidate/dual2d_decision.md`.
