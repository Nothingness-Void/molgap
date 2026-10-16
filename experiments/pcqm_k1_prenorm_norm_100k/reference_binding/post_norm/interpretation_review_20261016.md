# Normalization interpretation correction — 2026-10-16

This review supersedes the causal interpretation in both copies of
`attribution.md` and the pasted explanation. It preserves the original frozen
protocol, measured endpoints and NEGATIVE_UNDER_CONTRACT outcome for this exact
replacement. Those acceptance inputs are hash-bound; their bytes remain intact.
This is a source/retained-trace review, with no model execution or new role use.

## Supported result

The Kaggle2 fresh single-seed screen compared eighteen local post-residual BN
layers with eighteen local post-residual AdaRMSNorm layers, keeping the K1
training recipe fixed. BN selected live MAE is 0.141294461 eV at pass 40; Ada is
0.145184176 eV at pass 39. The candidate is 3.889715 meV worse, with the recorded
paired-row 95% interval [2.827877,4.949146]meV. It fails the frozen 0.5 meV
non-inferiority tolerance. This is valid evidence against direct substitution
under that recipe, not against all modern normalization or every K1 recipe.

Sources: [summary](summary.json), [protocol](../protocol.md), per-arm retained
`outputs/experiment/{arm}/canonical_trace.json`, and
`../training_recipe_{arm}.json`.

## Corrections to the explanation

| Earlier claim | Source-backed correction |
|---|---|
| No dropout; BN is the only stochastic regularizer | Frozen `qm9_neural_atom.py:222,240` sets dropout 0.05. `k1_ada_rmsnorm.py:43-52` only replaces norms; the shared factory does not zero dropout. Both train with dropout active. BN-noise effects were not isolated. |
| Train/dev subtraction proves severe overfitting | Train is the online pre-update, training-mode batch aggregate, with changing weights, dropout and BN batch statistics. Dev is fixed-checkpoint eval mode. Ada's quoted train value is pass 40, while selected dev is pass 39. These differences are not matched-state generalization gaps. |
| Development error collapses late | Ada dev improves0.148863 at pass 30 to 0.145184 at 39, then rises only0.046 meV at 40. BN also improves through 40. The traces do not establish a sustained late deterioration or a converged universal ceiling. |
| AdaRMSNorm necessarily destroys node magnitude | The implemented adaptive gain depends on the unnormalized input. With alpha=1,beta=0, its equal-epsilon numerator/denominator cancel and it can represent the identity. Whether this run learned useful magnitude information was not measured. |
| BN preserves physical energy ratios and has effectively exact statistics | BN's common channel shift/scale does not preserve arbitrary ratios; learned latent channels are not identified physical energies. Correlated graph nodes are not thousands of independent samples. No corresponding activation, chemical or effective-sample-size measurements were retained. |
| Slower execution is caused by extra kernel launches | Worker wall is4.62% longer. No isolated operator profile establishes how much is norm kernels, reduction, evaluation, checkpointing or other overhead. |
| Bootstrap establishes p<1e-4 and a universal mechanism | The recorded interval is conditional paired-row uncertainty for selected predictions. No p-value calculation or multi-seed variance estimate was retained. It cannot identify the causal mechanism. |

Training metric ownership is `k1_screen_training.py:511-549`,
`family_training_core.py:171`, and `family_training_session.py:282-310`.
Development eval ownership is `gptrans_execution_backend.py:152`.
Frozen source 1f627236 and checked-out bytes were compared for the K1 factory,
Ada module, shared factory, training core and session; they match.

## The important design mismatch

Local K1 executes
`h = Norm(x + Local(x)); y = Norm(h + MLP(h))`
(`qm9_neural_atom.py:87-93`). The addon preserves this post-norm residual path.
The global mixer still has its existing LayerNorms at lines 129, 139, 146; this
experiment did not replace the attention normalization system with PPGT.

PPGT uses pre-norm residual branches and a final normalization (equation 7).
Its AdaRMSN is explicitly designed to preserve magnitude when useful
(equation 3); its PCQM recipe also includes warmup, attention dropout and
stochastic depth. The local formula follows its adaptive-gain idea, but the
local experiment is not a reproduction of the whole architecture or recipe.
See [PPGT sections3.1/3.3 and appendixA table9](https://arxiv.org/html/2504.12588v4).
[Xiong et al.](https://arxiv.org/abs/2002.04745) independently establishes that
norm position changes optimization in Transformers; that does not predict a
K1 accuracy gain by itself.

Holding old hyperparameters fixed is useful for a direct-substitution screen.
It does not give each normalization an equally tuned recipe. Conversely,
jointly changing placement, optimizer and dropout would test a new package,
without identifying which change caused a gain.

## Relation to the completed 500K state diagnostic

The separately accepted [same-checkpoint review](D:/w/k1-clean-ema-gptrans-local-500k/experiments/pcqm_clean_k1_gptrans_local_500k/state_attribution/analysis_zh.md)
finds 0.109049479 eV for prefix-Clean EMA,0.105396973 for Raw EMA, and0.105419359
for uniform-Clean EMA at the same selected checkpoint. It supports sensitivity
to BN output/calibration state, not the claim that BN must be retained forever.
Its500K result does not change this 100K contract or establish a norm-replacement
training effect. Original cross-backend and historical-comparison limits apply.

## Decision scope and a better next question

Retain the exact direct-replacement negative result. Withdraw mandatory-BN,
zero-dropout, proven-overfitting and mandatory-extra-regularization conclusions.
Modern batch-independent normalization remains an unresolved candidate, not a
proven winner. EMA was tracked, but the retained traces have no EMA dev metrics;
the protocol's EMA-health objective was not answered.

A decision-relevant next design would complete an operator-by-placement table:

| Local residual placement | BN | AdaRMSNorm |
|---|---|---|
| Post-norm | Retained100K control | Retained100K negative |
| Pre-norm | New placement control | New placement plus adaptive norm |

Both new cells test changed architectures, not a repeat of the old baseline.
Freeze their common residual/final-norm policy, optimizer, exposure, roles,
selection state, runtime and initial shared tensors; bind qualified retained
controls before execution. Cross-job contrasts retain training-variance limits.
If recipe tuning follows, give both norm families the same prospective tuning
budget; do not assume that extra dropout is required. Record fixed-cohort,
same-checkpoint eval-mode train/dev metrics and actual EMA dev metrics if EMA
health is an objective. A short run can reject invalid/unstable code, not decide
terminal accuracy. No new training or 500K release is made by this review.

Parent reviewed the implementation/literature; an independent Luna read-only
audit checked trace units, selected passes, tail values and missing EMA/p-value
records. RML outcome and generated indexes are not reinterpreted or hand-edited.

## Verification

The requested `python -m molgap.research_memory check --frozen --portable`
command completed once in about 23.5 seconds. Its process exit code was not
captured by the invoking wrapper. The captured output reported 111 missing
local-artifact claims, all from older retained experiment records; no
`pcqm_k1_norm_ablation_100k` path appeared. The first three claims were:

- `experiments/family_100k_operator_profile/terminal_acceptance/outputs/experiment/gptrans_local/development_predictions.pt`
- `experiments/family_100k_operator_profile/terminal_acceptance/outputs/experiment/gptrans_local/last_checkpoint.pt`
- `experiments/family_100k_operator_profile/terminal_acceptance/outputs/experiment/gptrans_local/preflight.log`

Git status was unchanged by the check. No old terminal acceptance records or
`research_memory/derived/` files changed.
