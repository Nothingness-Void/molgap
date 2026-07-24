# Phase 8 Model Inventory And Repair Checklist

## Scope

This audit classifies model families rather than individual seeds and
checkpoints. Exact metrics remain in each experiment's decision record. The
machine-readable inventory is `model_inventory.csv`.

No new training or remote workload was submitted by this audit.

## Corrected Summary

It is incorrect to describe every post-500K model as a technical failure.

- Routed v4 is a confirmed positive production result.
- The fixed two-expert 1M ensemble and three-expert 2M ensemble are accuracy
  positive but fail the deployment-cost gate.
- Retention-B is positive on common, OOD, and P8-hard and is accepted as the
  general-model candidate under the new split objective.
- Several other models are valid specialists for OOD, broad residual, scaffold
  novelty, or PCQM-like chemistry.
- Retention-C was cancelled before a completed epoch; it has no quality result
  and must not be called negative.
- The PCQM Gap specialist is still running and has no accepted result.

Most "negative" decisions mean "not a single global default under the old
common + OOD + P8-hard + PCQM + cost contract", not "learned nothing".

## Model Checklist

### Accepted Or Reusable

- [x] **M00 Expansion500K v3 component**: retained as a stable component and
  compatibility loader.
- [x] **M01 routed-v4**: production general B3LYP predictor.
- [x] **M07 two-expert 1M ensemble**: confirmed accuracy gain; four GPS passes
  prevent default deployment.
- [x] **M09 three-expert 2M ensemble**: strongest common aggregate among the
  audited pure-2D candidates; six GPS passes and PCQM weakness prevent default
  deployment.
- [x] **M21 retention-B**: improves common/OOD/P8-hard and becomes the general
  branch under the split deployment objective.
- [ ] **M23 PCQM Gap specialist**: running; no result may be claimed yet.
- [x] **M24 conformer ensemble**: accuracy-positive, optional only because of
  approximately 6.8x inference cost.

### Data-Scale Models

- [x] **M02 original 1M fusion**: valid internal 1M reference, but external
  domain balance is worse than routed-v4.
- [x] **M04 repair-v2 1M**: OOD improvement did not compensate for P8-hard and
  PCQM regressions.
- [x] **M05 additive 1.5M**: OOD/PCQM-proxy signal increased while common and
  P8-hard retention collapsed.
- [x] **M06 broad residual 1.098M**: strong matched-domain specialist; poor
  global transfer.
- [x] **M08 exact-2M coverage**: OOD/scaffold novelty specialist; P8-hard
  regression blocks global use.
- [x] **M22 retention-C**: cancelled by decision, not evaluated.

### Compression, Routing, And MoE

- [x] **M10 distilled 2M GPS7**: one-pass compression did not retain common and
  P8-hard even though OOD/PCQM improved.
- [x] **M14 learned routers**: Oracle gain exists, but pre-expert descriptors,
  disagreement, embeddings, and uncertainty do not predict the winner with
  sufficient reliability.
- [x] **M15 late blends**: bounded fixed blends produced sub-threshold gains;
  learned blends regressed.
- [x] **M16 heterogeneous MoE**: the SchNet expert was too weak and router
  gains were seed-unstable.
- [x] **M17 static dual-2D**: internal complementarity did not transfer across
  all seeds and domains.

### Fusion And Structural Heads

- [x] **M03 replay-weighted 1M head**: changing only the head cannot repair
  encoder-level representation shift.
- [x] **M11 2M-2D + 1M-3D fusion**: encoders were trained on different data
  mixtures; late fusion could not restore alignment.
- [x] **M12 exact-2M encoder transplant**: controlled three-seed evidence shows
  that new GPS representations are not drop-in replacements for the fixed
  500K SchNet branch.
- [x] **M13 copied Gap<4 route**: the route was representation-specific; on 1M
  the dual branch is broadly stronger and should not be conditionally hidden.
- [x] **M18 physics-consistent head**: correct output algebra is not equivalent
  to lower external error.
- [x] **M19 structural adaptor**: the proposed ring/conjugation mechanism was
  absent in the worst residual decile.
- [x] **M20 legacy layer/head probes**: correlated intermediate features added
  complexity without transferable supervision.

## Why Global Promotion Repeatedly Failed

### F1. Targeted 500K dilution

The immutable targeted 500K represented 50% of 1M, 33.3% of 1.5M, and 25% of
2M. The original general top-up was much smaller, less aromatic, less flexible,
and higher-Gap than the targeted base. Increasing row count changed the
optimization distribution.

### F2. Planned replay was absent

Completed 1M/1.5M/2M encoder runs used uniform sampling even when experiment
plans specified replay. Retention-B is the first direct correction and its
common/OOD/P8-hard gains confirm that this was a causal problem.

### F3. Internal validation was not stationary

Each random split followed its own enlarged data mixture. Internal validation
improved as the mixture became easier, while fixed P8-hard retention regressed.
Internal MAE was therefore not a valid cross-dataset promotion metric.

### F4. Specialist gains point in opposite directions

Coverage-2M improves OOD but damages P8-hard. Broad-residual training strongly
improves its matched domain but not the main evaluation distribution. Absolute
residual correlations around 0.99 leave little benefit for unconditional
averaging.

### F5. Routing rules did not transfer

The production Gap<4 rule captures a specific 500K GPS/SchNet relationship.
After encoder retraining, branch strength and error regions changed. Learned
routers also failed because pre-expert features did not reliably predict
per-molecule gain.

### F6. 2D/3D representations were misaligned

The 2M 2D encoder and 1M 3D encoder saw different training mixtures. Wider
fusion added correlated inputs, projected them into the same 192-dimensional
bottleneck, and had no new aligned supervision.

### F7. Compression optimized the wrong retention target

The distilled student matched an expensive ensemble on the exact-2M split, but
the teacher itself encoded domain tradeoffs. Better internal teacher matching
did not guarantee common/P8-hard retention.

### F8. Deployment gates mixed accuracy and cost

The two- and three-expert ensembles are accuracy-positive. They were not
promoted because they require four or six GPS passes. These are cost
rejections, not model-quality failures.

### F9. The old product objective was over-coupled

One model was required to win common, OOD, P8-hard, PCQM, and inference cost.
PCQM supplies only Gap labels and has a distinct chemical distribution. The
new explicit general-model + PCQM-specialist split removes this artificial
conflict without hiding a metric regression.

## Repair Checklist

### P0. Freeze contracts

- [x] Keep routed-v4 unchanged as production until a complete deployment gate.
- [x] Preserve the raw 2M corpus; never overwrite it during repair.
- [x] Separate the general B3LYP contract from the PCQM Gap-specialist contract.
- [ ] Record one immutable identity/split manifest for every new comparison.
- [ ] Define cost limits before architecture selection.

### P1. Repair the 2M data representation

- [ ] Build one row-level ledger containing source round, CID, canonical SMILES,
  scaffold, Gap, MW, heavy atoms, aromatic rings, rotatable bonds, elements,
  fragmentation, charge, and quality flags.
- [ ] Reconcile every unused Kaggle acquisition round against the historical
  inventory by CID and canonical SMILES.
- [ ] Mark invalid, radical, disconnected, noble-gas, extreme-label, and
  conflicting rows; do not silently mix them into the B3LYP three-target set.
- [ ] Quantify joint buckets, not only marginal histograms: Gap x MW x
  aromaticity x flexibility x scaffold density.
- [ ] Keep the targeted 500K immutable.
- [ ] Downsample or cap highly redundant small/high-Gap/simple scaffolds through
  a sampling manifest instead of deleting source rows.
- [ ] Select replacement candidates for the under-covered Gap 2-4 eV,
  high-aromatic, MW>700, rotatable>10, heavy-atoms 35-50, donor-acceptor, and
  rare-element regions.
- [ ] Build a fixed-size repaired-2M manifest before materializing another CSV.

### P2. Controlled GPS7 gate

- [x] A: uniform exact-2M result exists.
- [x] B: exact-2M with 50% targeted replay exists and improves
  common/OOD/P8-hard.
- [ ] D: repaired-2M with the exact B initialization, replay, split, optimizer,
  epochs, and seed.
- [ ] Compare D against B, not only against routed-v4.
- [ ] Require common regression <=0.0005 eV, at least 0.001 eV improvement on
  OOD or P8-hard, and <=0.0005 eV regression on the other domain.
- [ ] Audit worst molecular buckets and paired confidence intervals.
- [ ] Repeat only a passing D configuration across three seeds.

### P3. Architecture after data passes

- [ ] Train GPS9 only after repaired-2M GPS7 passes.
- [ ] Recompute OOF branch gains; never copy the old Gap<4 threshold.
- [ ] Require Router-predictable gain before training another Router.
- [ ] Retrain 3D on the same molecule mixture and split before fusion.
- [ ] Keep the accepted prediction as an identity path and learn only a
  bounded target-specific residual correction.
- [ ] Reject any fusion that only widens correlated inputs while retaining the
  same bottleneck and supervision.

### P4. Split deployment

- [x] General branch candidate: retention-B.
- [ ] PCQM branch: accept only after the 95,909-row clean-train specialist
  passes scaffold validation and official-valid local evaluation.
- [ ] Route by requested task/label domain, not by an unreliable molecule-level
  learned Router.
- [ ] Keep HOMO/LUMO on the general B3LYP branch; the PCQM specialist is
  Gap-only.
- [ ] Revalidate Delta/UQ only after final branch checkpoints are fixed.

## Immediate Order

1. Finish and evaluate the PCQM specialist already in progress.
2. Build the local 2M row-level ledger and reconcile unused acquisition pools.
3. Produce a repaired-2M sampling manifest without retraining.
4. Review the exact before/after bucket counts.
5. Run only the controlled D-versus-B GPS7 experiment.

Do not start GPS9, MoE, Router, 3D, or fusion before step 5 passes.
