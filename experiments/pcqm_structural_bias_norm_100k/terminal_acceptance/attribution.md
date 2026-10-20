# Terminal attribution

Dated2026-10-20 JST; retained prediction/trace/profile arithmetic only.
The [decision](decision.md) preserves the frozen3meV gate and
[analysis](analysis_summary.json) preserves bootstrap inputs and semantics.
No model execution, additional training or protected-role evaluation occurred
during local acceptance.

## K1: direct post-residual LN substitution

The selected Live candidate is4.711meV worse than matched BN, with paired-row
95% interval[+3.640,+5.756]meV. Both arms consumed the same31240 updates and
3998720 presentations. Native and full-resume checks passed, contradicting an
interruption, unequal scheduled exposure or accidental EMA endpoint explanation.

BN's final10/5-pass development improvements are4.013/0.968meV. LN's are
5.269/-0.037meV; LN selects pass36 and ends0.353meV worse at pass40. The final
LN primary curve is approximately flat while BN still improves. This does not
establish that extending LN would recover its deficit. Absolute convergence,
underfit and exposure sufficiency are not identified by a single scheduled run.

Final online train MAE scales to0.098231eV for BN and0.095552eV for LN.
LN fits the changing-weight training stream better while its development score
is worse. That is compatible with a transfer/regularization difference, not
proof of overfitting: these are not same-checkpoint, fixed-cohort training
evaluations. Activation statistics, gradient conditioning, and a uniquely causal
norm mechanism remain insufficient_evidence. The experiment isolates the
declared post-residual replacement under the frozen common recipe; it is not
independent hyperparameter tuning or a general verdict about modern norms.

LN worsens median,95th and99th absolute-error quantiles; its mean signed error
is-3.397meV versus BN+3.868meV. These are descriptive row summaries, not a
new calibration intervention. No post-hoc calibration or checkpoint inference
was run. Ring/size/connectivity cohorts are not present in the retained
prediction bundles and were not fabricated or reconstructed from new data.

## GPTrans: resistance bias on degree-scaled BondLocal

Selected EMA improves1.119meV, paired-row95% interval[-2.057,-0.161]meV.
The point gain misses3meV. Exactly25052 rows improve and24948 worsen, so the
signal is not broad per-row dominance. Median absolute error slightly worsens
(0.092367 to0.092847eV), while95th/99th percentiles improve
(0.436451/0.776606 to0.428701/0.757917eV). This is compatible with heterogeneous
tail-error benefit; cohort-specific or molecular causal attribution is absent.

BondLocal selects pass52; resistance selects pass45 and ends0.541meV worse
than its selected endpoint. Final10-pass changes are+0.073meV improvement for
BondLocal and-0.021meV for resistance. Neither retained EMA tail supplies
evidence for an exposure-shortage rescue. Both completed46860 updates and
5998080 presentations. Online normalized training metrics use Live weights,
while primary development uses EMA; their difference is not a generalization gap.

## Actual cost and profiler scope

Measured invocation wall: K1 BN5670.911s, LN5640.866s (LN0.53% shorter);
GPTrans BondLocal9431.775s, resistance9554.839s (resistance1.30% longer).
Unprofiled complete-pass training-span medians are120.532/119.721s for
K1 and132.746/134.004s for GPTrans. The native16-update profiles completed
in every arm. Summaries retain no pinv/SVD/eigh entry in their truncated top
operator lists; that absence does not quantify zero pinv cost. End-to-end
measurements support a small observed overhead, not an isolated causal estimate.

Shared allocation ledgers separately record K1 wall5729.812s and
11459.624 T4-device seconds; GPTrans wall9626.834s and19253.668 T4-device
seconds. Total is8.531469969 T4-device hours. The Python observation window
includes idle allocation; queue, pre-Python provisioning, post-exit release,
account-quota debits and CPU consumption remain unknown, not zero.

## Evidence publication and disposition

The existing frozen writer publishes null canonical learning-rate, checkpoint
identity and timing histories in these modes. Standard epoch logging is not
canonical checkpoint identity. Preserve those bytes and explicit strict-replay
exclusions. The first retrieval manifest also omitted its required trace
context companion; four small version-specific JSON companions were retrieved
and independently pinned before mechanical checks passed. No raw training
trace was retrofitted and no baseline was retrained for acceptance.

Close both candidates negative under the material contract and retain the
resistance result as a weak numerical signal. Mechanism discrimination would
need a separately scoped fixed-checkpoint/cohort diagnostic or a newly justified
recipe question; none is authorized or launched here. Original failed attempts
provide bootstrap attribution only and must not be combined with training cost
or exposure as if they were successful scientific seeds.
