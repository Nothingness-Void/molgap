# PCQM 100K-to-500K scale-transfer attribution

## Decision

The repository does not yet support the broad claim that most positive 100K
architectures fail at 500K. It contains only two direct mechanism transfers
under sufficiently controlled contracts, and both failed to retain a material
gain:

| Mechanism | 100K gain | 500K evidence | Disposition |
|---|---:|---:|---|
| K1 PairToken | `+0.003044 eV` | `+0.000556 eV` | retained 18.3%; negative under gate |
| GPTrans Pair PreNorm | `+0.003125 eV` | `-0.001277 eV` mean over matched epochs 20-36 | sign reversal; stopped for cost |

The observed funnel failure is real: zero of the two known 100K winners
retained the required `0.003 eV` gain. The sample is too small to infer a
general law, but it is sufficient to reject immediate 100K-winner-to-full-500K
promotion as the default workflow.

## Why the gains disappeared

The strongest explanation is **selection and horizon fragility**, not that the
larger dataset somehow damages every new module.

1. Both winners barely crossed the gate. PairToken exceeded it by only
   `0.000044 eV`; Pair PreNorm by `0.000125 eV`. Repeatedly choosing the best
   result on one fixed development role makes such boundary wins vulnerable to
   selection bias. Row bootstrap cannot measure training-seed variation.
2. Both selected their last available 100K epoch: 39/40 and 59/60. Their
   promotion evidence therefore measured one finite-horizon trajectory point,
   not a stable asymptotic advantage.
3. The 100K and 500K contracts change both the training population and the
   development cohort. The 100K role is rows 100000-150000; the 500K role is
   rows 500000-550000. A changed gain is not a pure data-scale effect.
4. Optimization time changes sharply. PairToken moves from 31,240 steps and
   4.0M presentations to 234,360 steps and 30.0M presentations. The added
   relation bottleneck can help sample efficiency at 100K while becoming
   redundant after the base K1 path sees more data and updates. PairToken's
   81.7% gain loss is consistent with this mechanism, but existing evidence
   cannot prove it without matched-prefix traces.

The historical full-scale failures have additional confounders and must not be
used as clean confirmation. The first EdgeState full run changed to a lossy
feature contract and was underconverged. The K1/GPTrans full runs used only
20M presentations, while the converged EdgeState checkpoint selected epoch 30,
roughly 101.36M presentations. Their official-validation ranking is therefore
important operational evidence, but not a controlled architecture scaling
result while the convergence chains remain unresolved.

This interpretation is consistent with published warnings that repeated model
selection can overfit a finite validation criterion and that learning curves
must be measured across data scales rather than assuming fixed rankings:
[Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) and
[Hestness et al. (2017)](https://arxiv.org/abs/1712.00409).

## Funnel repair

Do not lower the material gate and do not launch another architecture merely
to keep compute occupied. Repair the evidence path in this order:

1. Recover the existing PairToken 500K trace plus its exact K1 reference
   payload. RML currently excludes the only complete transfer from backtesting
   because that reference is missing.
2. Freeze one transfer-confirmation development role shared by future 100K and
   500K screens and disjoint from both training prefixes. Keep the existing
   role for continuity, but do not use it alone to authorize scale-up.
3. Record candidate-versus-reference gain at fixed optimizer-step and sample-
   presentation checkpoints. Require a favorable terminal tail, not only two
   independently selected best epochs.
4. Treat boundary wins as provisional. A candidate within the unmeasured
   training-variance band needs one reusable second-seed confirmation before a
   500K allocation.
5. Run future 500K bridges in frozen stages. Compare at matched presentation
   prefixes first; continue to the next stage only if the gain persists. Exact
   stopping thresholds must be activated only after RML trace backtesting, not
   invented from these two cases.

## Current experiment routing

The already submitted K1 Functional-Group Token 100K screen remains the only
active model question. Its terminal result must be accepted and attributed
before any 500K action. This analysis authorizes no additional compute and no
protected-role access.

