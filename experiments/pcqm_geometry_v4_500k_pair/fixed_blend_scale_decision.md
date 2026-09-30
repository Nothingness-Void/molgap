# Fixed-blend scale review, 2026-09-30

Decision: **do not release full-scale geometry training from this evidence**.
The original geometry-versus-own-reference point gates remain failed. The
predeclared geometry blend's complementarity nomination remains valid, but
the recovered pure-2D blend supplies the more relevant comparator for deciding
whether geometry adds enough value to justify a new full run.

## Recovered reference and analysis

The accepted K1 predictions were found in the historical desktop artifact
checkout `D:/文档/molgap-exp/molgap-500k-v4-evidence/`, rather than the primary
checkout or the Codex-managed worktrees previously searched. Their SHA256 is
exactly `68fba785a0b028a8445fe8d94d348df2c3a8d7d8caab708acb574d13cac9831b`;
the corresponding accepted stage-manifest hash also matches. The previous
missing-reference assessment was an incomplete inventory, not lost evidence.
Recovery copied only the selected predictions and manifest. The recovery
receipt is under `platforms/_records/kaggle/training/pcqm_500k_v4_evidence_reference_predictions/`.

`compare_fixed_blends.py` reuses the owning matched-V4 prediction loader,
alignment checks, bootstrap and atomic IO. Four immutable selected prediction
artifacts match their accepted hashes, exactly align source indices
500000–549999, have finite values and identical targets. No model execution,
remote download, training or additional role was introduced. This is a
supplementary review of the existing geometry question on the already used
development cohort; hybrid substitutions are post-hoc and non-promotional.

| Fixed 50:50 combination | Development MAE, eV |
| --- | ---: |
| Original pure-2D GPTrans + original pure-2D K1 | 0.100655615 |
| GPTrans distance + K1 distance-angle | 0.100290090 |
| GPTrans distance + original pure-2D K1 | 0.100461803 |
| Original pure-2D GPTrans + K1 distance-angle | 0.100396208 |

The geometry blend improves **0.365530 meV** over the original fixed blend.
Its paired-row 95% percentile interval is **[-0.033628, 0.767960] meV** and
crosses zero (10,000 draws, seed 20260917). It has lower absolute error on
50.356% of rows. A reliable blend-over-blend advantage is not established.

The original pure-2D pair already improves 4.204258 meV over its better
component. Thus the 4.573 meV improvement of the geometry blend over its own
better component largely reflects existing architecture complementarity.
Crediting that entire gain to geometry would overstate the evidence.

K1 geometry versus its recovered pure-2D reference changes MAE by +0.003281
meV, with an improvement interval [-0.568534, 0.567195] meV. GPTrans distance
improves 0.857126 meV, interval [0.340818, 1.377350] meV, a supported small
endpoint effect below its frozen 3 meV gate. Each hybrid substitution also
has an interval crossing zero. There is no demonstrated material blend gain.

The 3 meV value here is the original individual-arm materiality context;
no new blend-versus-blend threshold was retrospectively preregistered. The
no-full decision follows the small observed increment, its uncertainty, and
duplicate architecture-complementarity evidence. Row resampling does not
measure training-seed uncertainty or guarantee scale transfer. The historical
three-model crossfit result used a different blend procedure and remains
exploratory; it is not a strict rank against this fixed blend.

## Scientific interpretation and operating disposition

These are full 60-epoch results under the frozen V4 exposure. The evidence
supports a small GPTrans distance gain and no detected K1 geometry gain.
It does not establish a causal explanation for their limited benefit, prove
geometry universally useless, or justify longer training. Both full and
pure-2D fusion histories already show that complementary 500K errors alone
do not ensure improvement over the accepted full-scale EdgeState reference.

Use the existing full-scale reference and the accepted pure-2D pair for
future comparisons. Reopening full geometry training requires materially
new, decision-relevant evidence or an explicit different scientific question
with its own prospective contract. This review releases no successor.

The K1 reference-prediction recovery blocker is resolved. The pre-existing
immutable terminal RML transactions and scientific observations are preserved.
Their replay exclusions are not removed by this saved-prediction comparison.
Independent reference/readiness binding and the supported correction path for
the 0.971-second GPTrans cost discrepancy remain bookkeeping work. Neither
changes the measured blend-over-blend result or justifies another training run.
Keep the owning branch for that technical reconciliation and final Git routing.
