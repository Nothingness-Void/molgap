# Post-chemistry-local route review — 2026-09-28

This is an evidence-selection review, not a training result, a new benchmark
role, or a compute release. The two chemistry-local arms are already closed by
their [terminal decision](../pcqm_k1_chem_local_100k/decision.md). Both are
strict/replay-ready negative results. No seed, fixed500K audit, or scale-up is
released by their protocol.

## What the accepted runs now rule out

- More K1 local bond capacity is not a sufficient hypothesis. Both equal-size
  layer-6 colorings fitted the training role better and generalized worse.
- Repeated small original-role gains cannot be treated as scale evidence. The
  [relation-resolution audit](../pcqm_k1_relation_resolution_100k/audit/decision.md),
  [topology audit](../pcqm_k1_portability_dual_100k/audit/decision.md), and
  [PairToken cross-scale diagnostic](../pcqm_k1_cross_scale_frozen/decision.md)
  all observed attenuation or reversal with frozen 100K weights on different
  internal molecules. This cannot be attributed solely to 500K training steps.
- Post-hoc K1-error strata are not inference-time selectors. The existing
  [molecular-router audit](../pcqm_molecular_router_audit/decision.md) found
  large label Oracle headroom but little winner-identification signal from
  structure or prediction disagreement.

## Read-only exploratory checks, not formal claims

The already accepted 50,000-row aligned SMILES/prediction matrix from the
router audit (SHA-256 `c52e944def35d8d9ccb41011c9cf820182702c1a9006189b6fc488397169edbb`)
was joined by exact source index and target to the two accepted chemistry-local
payloads. No model was trained or inferred and no official role was read.

- Inference-visible atom count, ring count, aromatic/conjugated fractions,
  rotatable bonds, heteroatom count, K1 prediction and absolute model
  disagreement each gave only about `0.50–0.51` single-feature AUC for
  identifying which chemistry-local arm beats K1. This is an in-sample,
  multiple-inspection diagnostic, not an independently validated Router gate.
- K1's MAE on the 3,958 RDKit-identified radical-containing molecules was
  `0.238932 eV`, versus `0.141374 eV` overall. The strongest inspected
  radical-subgroup gain was `0.008130 eV` for the globally negative sparse
  triplet candidate. Even a perfect radical-only switch to that particular
  checkpoint would contribute only about `0.000644 eV` to the overall mean,
  before inference cost or transfer uncertainty. Rare Si/P rows were also
  difficult but too small to justify an untested extra encoder.
- A crude, **non-learned** adjacency spectrum of conjugated components could
  be formed for 45,096/50,000 molecules. Its rank correlation with the Gap
  label was about `0.361`, but with K1's signed residual only `-0.016`.
  This is not evidence of an incremental Hückel-feature advantage; it is not
  a chemically validated orbital calculation.
- Fixed equal averaging of the two negative arms with K1 improved the reused
  development-role MAE, but an older K1 + PairToken two-model average already
  matched or bettered the three-model result. Neither this post-hoc average
  nor a label Oracle establishes a new chemistry-specific mechanism.

## External architecture priors screened, not transplanted

- [MetaGIN](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-024-3784-y)
  reports a pure-2D PCQM4Mv2 result with a 3-hop/MetaFormer backbone, but its
  reported 8.87M-parameter full-scale recipe is not the frozen 3.66M K1
  100K/40-epoch contract. The public [source snapshot](https://github.com/xwxztq/MetaGIN/tree/32154dbce8c3aa8268eb973288dd512a5774536d)
  at `32154dbce8c3aa8268eb973288dd512a5774536d` confirms a distinct
  bond/2-hop/3-hop backbone whose forward pass does not consume `pos_3d`.
  However, the published `main.py` imports dataset symbols that are commented
  out in `data.py`, constructs official-valid/test-dev loaders, and specifies
  batch 256, Adan, and 12 periods of 12 epochs. Its preprocessing preferentially
  constructs training graphs from an SDF and other graphs from SMILES. None of
  that code can be run as-is under the fixed cross-platform 2D data identity,
  protected-role boundary, or V5 batch-128 contract. Local path and
  extra-local-capacity evidence is weak or negative; a shrunken K1 add-on
  would not reproduce the paper. A faithful, role-safe independent backbone
  would require a separate source/graph/runtime preflight before any GPU gate.
- [CTNN](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cffcd7e5c10755a3439e812f3beffde2-Abstract-Conference.html)
  offers a distinct canonical tree-cover backbone, but the reported sparse
  molecular results are classification rather than PCQM Gap regression.
  Its graph canonicalization and runtime require independent preflight.

**Decision:** neither paper currently clears an evidence-backed GPU release
threshold. The next useful question is not another local-width/selector/gate
variant. It is whether a genuinely new 2D information source explains K1
residuals on molecules outside the repeatedly used original-development role,
under a predeclared no-training audit. Only then freeze a different backbone
or information-flow experiment. Do not treat this review as an RML terminal or
as permission to open official validation/test roles.
