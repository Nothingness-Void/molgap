# Work Tracks

This file answers one question: **what do Track A, B, and C mean?**

| Track | Purpose | Promotion rule | Primary location |
|---|---|---|---|
| **A - Production** | Build the general PubChemQC/B3LYP model and the shipped database | May change the production registry only after fixed external acceptance | `production/` and production-directed experiments |
| **B - Leaderboard** | Maximize PCQM Gap accuracy for benchmark submission | Remains a task-routed specialist; never replaces Track A without a separate production gate | `experiments/pcqm_gine_expert/`, `experiments/pcqm_route_b/` |
| **C - Discovery** | Use QM9 and bounded transfer screens to eliminate or discover architectures cheaply | A result must transfer to the target domain before entering Track A or B | `experiments/qm9_architecture/`, `experiments/pubchemqc100k_architecture/` |

The flow is directional:

```text
Track C discovery -> target-domain validation -> Track A or Track B
Track A production != Track B leaderboard
```

Names such as `route_b_fusion.py`, `pcqm_route_b.py`, and existing
`route_b_*` result files are stable implementation and artifact identifiers
created before this taxonomy. Do not interpret them as track ownership and do
not rename them while active remote jobs or reproducibility records depend on
them. In prose, use **bounded 2D+3D fusion** for that architecture.

Live status is in `CURRENT_STATE.md`; priority is in `ROADMAP.md`.
