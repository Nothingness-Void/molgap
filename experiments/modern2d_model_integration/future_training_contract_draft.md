# Future Modern2D contract: non-executable draft

Status: NOT_AUTHORIZED_FOR_SCIENTIFIC_TRAINING. This document is not Spec,
trajectory, runtime certificate, schedule, submit input or a training release.
No automatic action follows from CPU integration or reference numerical parity.

## Frozen structural candidates

- PPGT main: deep_narrow16x256,8heads, pair128, stem8/MLP512, RRWP24/SPE10,
  expansion2, AdaRMS epsilon1e-8, phi enabled, sum/head3. Reserve: shallow4x512.
- ESA main: M-S-M-S-M-P,192/8heads, LN-pre epsilon1e-8, gatedMLP384/2layers,
  seeds32, graph256, self-enabled mask. Reserve: M-M-S-P,256/8,MLP512.
- CPU defaults disable dropout/SD/EMA; these are not selected native training
  hyperparameters. Any later regularization/EMA change is frozen explicitly.

## Required releases, in order

1. Resolve PPGT independent fidelity: verified author source or separately
   authorized original ZINC-12K reproduction. Explain the Table9 parameter
   mismatch; a bad score before this gate is IMPLEMENTATION_UNVERIFIED, not a
   negative judgment on the architecture. ESA-small mask/OGB input adaptations
   remain separate from the accepted author-reference configuration.
2. Review a real indexed V2 input loader and exact accepted manifest/chunk maps.
   Freeze derivation source/config, topology/source/order/role history and any
   materialized features. No new split, graph rebuild, old-dev substitution or
   bulk dense RRWP cache. Resolve ESA empty-edge incidence on authorized
   train topology before deciding whether explicit rejection is feasible.
3. Authorized train-topology size distribution and bounded CPU feature timing
   choose on-the-fly versus derived shards. Preserve immutable base graphs.
4. Separate native preflight authority: exact platform/account/assets/runtime,
   physical BS128, FP32/TF32-off, full allocation cost, memory/throughput,
   same-weight numerical and optimizer-inclusive resume checks. Current CPU
   per-graph loops and synthetic caps are not native throughput qualifications.
5. Freeze a prospective comparison with symmetric tuning budgets for new
   families and K1/GPTrans: normalization/loss/optimizer/LR/warmup/steps,
   selection/regularization/EMA, common roles, source and total compute ceiling.
   LR/step values are UNSET, not silently inherited from dissimilar papers.
6. One paired seed42 screen by default. 100K is diagnostic, never an automatic
   veto on500K. Use step-aligned curves and uncertainty/cost; no legacy STQS or
   uncalibrated automatic early stopping. Additional seeds or500K require new
   user compute authority; no automatic promotion from a promising curve.
7. Desktop owns full-scale/official work. Independent Audit access is one-time,
   separately authorized, and only after candidate selection is frozen.

## Open limitations

No real indexed inputs, mirror publication, parameter-table reconciliation,
native certificate, scientific Spec, MAE, GPU activation peak or train cost is
established here. Paper hyperparameters/results are references, not transferable
performance promises. The accepted V5/RML/Cross-Scale owners remain unchanged.
