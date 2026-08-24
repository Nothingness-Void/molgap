# Resource-Bounded Architecture Refresh

This Track C experiment asks which from-scratch 2D or integrated 2D+3D
architecture can improve the frozen repaired-2M production comparator while
keeping one full training run below 12 hours on the available platforms.

## Status

The feasibility review, prior-fusion failure audit, and all PubChemQC 100K
rounds are complete. Three-output RWSE16 Structural GPS9 passed its first gate;
persistent EdgeState Structural GPS then improved on that model in all three
seeds and became the sole repaired-2M scale-up candidate. Gap-only supervision
regressed, normalized/gated RWSE did not beat the accepted three-output model,
and the GatedGCN branch failed its direction-consistency gate. The full
repaired-2M EdgeState run has not started because its immutable input and
one-epoch timing contract remain incomplete. The separate conservative 2D+3D
head and its Colab runner are ready, and the CPU-only IMS staging payload has
passed local replay acceptance; its model training has not started. The
production registry remains unchanged. Exact decisions are in `decision.md`
and the per-question directories under `results/`; remote provenance is in
`STATUS.md`.

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
- `accept_paired_kaggle_screen.py` - read-only paired artifact and prediction
  acceptance across both three-seed Kaggle runs.
- `accept_gap_rwse_screen.py` - read-only Gap-column comparison across the old
  three-output model and both new Gap-only variants.
- `accept_gated_feasibility.py` - strict one-seed GatedGCN precision and
  runtime gate against the accepted Structural GPS seed.
- `accept_gated_multiseed.py` - strict three-seed acceptance across separately
  packaged confirmation kernels, including ensemble and paired deltas.
- `accept_edge_state_multiseed.py` - the corresponding strict persistent-edge
  three-seed gate.
- `preflight_conservative_fusion.py` - accepted-input and exact-identity safety
  check for the P1 2D+3D repair.
- `build_conservative_fusion_payload.py` - exact-alignment compact payload for
  the Drive-backed P1 run.
- `platforms/colab/conservative_2d3d_fusion/` - A100 notebook, immutable wheel,
  upload layout, and per-epoch resume contract for the P1 run.
- `platforms/scnet/resource_bounded_architecture/` - SCNet preflight and the
  six-task GPS9/Structural GPS array.
- `platforms/kaggle/training/resource_bounded_architecture/` - allocation-
  blocked fallback with two independent three-seed GPU kernels.
