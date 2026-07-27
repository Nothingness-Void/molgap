# Phase 8 Structural GPS Adapter -- Pre-registration

Date: 2026-07-17

## Hypothesis

The remaining routed-v4 B3LYP Gap residual is enriched in molecules whose ring
and conjugation topology is not explicit in the current 9-dimensional atom and
4-dimensional bond input. A small residual adaptor over frozen GPS node states
may improve those cases without changing the B3LYP labels, ETKDG method, data
volume, or the SchNet component.

## Fixed comparator and scope

- Comparator: `phase8_routed_dualgps_hybrid` (routed-v4), recorded at commit
  `4718f541366b7eefe1c4671b0a41f7d9fa044d33`.
- Primary target: B3LYP Kohn-Sham Gap MAE (eV). HOMO and LUMO remain required
  diagnostics.
- G0 input: the stored routed-v4 common-eval prediction table. No model fitting,
  checkpoint update, external-label access, or sealed-set access is allowed.
- G0 features: atom-in-ring, smallest ring size, atom ring-membership count,
  fused-ring excess, bond-in-ring, and RDKit bond conjugation. No cross-molecule
  ring-system identifier or descriptor-only fusion feature is allowed.

## G0 decision rule

The implementation must be deterministic from canonical SMILES, finite for all
valid 2D graphs, and have complete coverage of valid rows. The audit reports
feature prevalence in absolute Gap-error deciles of routed-v4.

G1 is permitted only when at least one non-trivial topology feature is enriched
in the highest-error decile versus the full evaluated population and that signal
is chemically interpretable. A flat or low-coverage audit archives this route;
it does not justify a second standalone 2D expert, static blending, a router,
or a descriptor fusion retry.

## Pre-declared G1/G2 protocol if G0 passes

- Development: fixed 30k scaffold-disjoint pilot; seeds `42`, `43`, `44`.
- Trainable parameters: structural residual adaptor and replacement dual-GPS
  fusion head only. Existing GPS and SchNet checkpoints remain frozen.
- Identity requirement: at zero adaptor update, the candidate must reproduce the
  frozen base representation exactly.
- External blocks, evaluated once with frozen choices: common all, OOD-1000,
  P8 targeted hard, and PCQM-like proxy.
- Promotion: at least `-0.001 eV` common Gap improvement, with no seed/block
  Gap regression above `+0.0003 eV`; report HOMO/LUMO, latency, and structural
  slice results.
- Any failed external block archives the independent-topology-enhancement family.

No sealed set will be opened for this experiment.
