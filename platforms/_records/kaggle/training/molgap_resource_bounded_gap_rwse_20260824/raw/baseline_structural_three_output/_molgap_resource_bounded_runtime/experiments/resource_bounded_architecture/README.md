# Resource-Bounded Architecture Refresh

This Track C experiment asks which from-scratch 2D or integrated 2D+3D
architecture can improve the frozen repaired-2M production comparator while
keeping one full training run below 12 hours on the available platforms.

## Status

The feasibility review and prior-fusion failure audit are complete. No candidate
has a training result yet, no remote run is authorized by this record alone,
and the production registry is unchanged.

Read in this order:

1. `decision.md` - dated feasibility conclusion, shortlist, gates, and sources.
2. `fusion_failure_audit.json` - machine-readable external saturation evidence
   for the rejected late 2D+3D residual design.
3. `ROADMAP.md` - live task order and run authorization boundary.
4. `CURRENT_STATE.md` - live production comparator and experiment status.

Reusable encoder logic belongs in `src/molgap/`; future command-line adapters
and compact results belong in this directory. Existing QM9, PubChemQC 100K,
repaired-2M, and conformer evidence must be referenced rather than copied or
rerun.

Implementation entrypoints:

- `build_rwse_cache.py` - resumable RWSE graph enrichment.
- `preflight.py` - aligned-cache and accelerator forward/backward contract.
- `accept_screen_run.py` - strict per-seed artifact acceptance.
- `platforms/scnet/resource_bounded_architecture/` - SCNet preflight and the
  six-task GPS9/Structural GPS array.
