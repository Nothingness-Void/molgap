# Phase 8 Local Readiness

This directory answers one question: which local prerequisites are ready
before the active remote outputs arrive?

| Deliverable | State | Authority |
|---|---|---|
| Three-seed equal ensemble | Complete; accuracy mode only | `../repaired_2m/retention_d_three_seed_equal_ensemble_decision.md` |
| GPS7/GPS9 OOF folds | Frozen; jobs prepared but not submitted | `../repaired_2m/gps7_gps9_oof/manifest.json` |
| Route B Fusion package | Ready; minimal/cost/precision presets | `../../../src/molgap/route_b_fusion.py` |
| Route B SchNet acceptance | Ready; both branches required | `../../../src/molgap/artifact_acceptance.py` |
| Repaired-2M 3D acceptance | Ready; expects 100 immutable shards | `../../../src/molgap/artifact_acceptance.py` |
| Model database | Updated; production/candidate/expert roles separated | `../model_inventory_audit/model_inventory.csv` |
| Report table | Generated with blank Route B cells | `../reporting/model_comparison.md` |

## Safety Locks

- Router training remains forbidden until genuine held-out OOF predictions
  populate the frozen gain-label contract.
- Full repaired-2M SchNet training remains forbidden.
- The sealed 20K remains unopened.
- The production registry/default remains unchanged.
- No remote job is submitted by these local preparation scripts.
