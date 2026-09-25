# PCQM scale-transfer postmortem, 2026-09-25

## Decision

`NO_TRAIN`: the accepted evidence supports a matched-500K **early optimization
advantage that mostly erodes**, but does not isolate the causal reason for
100K-to-500K transfer or prove an architecture-only 500K-to-full rank
reversal. Keep the accepted converged EdgeState checkpoint as the operational
full-scale reference. Do not promote another 100K winner or launch a full run
from the old STQS score. No new evaluation role, inference, or training was
used in this audit.

The previous `pcqm_scale_transfer_attribution` report had two direct transfer
cases and reasonably rejected immediate 100K-to-full promotion. The later
`pcqm_scale_transfer_diagnostic` claim that exposure mapping *proves*
representation collapse is not supported: exposure mapping alone has no
representation measurement. Its projected joint 500K gain of at least
`0.0078 eV` failed against the accepted `0.002056 eV` gain, a shortfall of
`0.005744 eV`. STQS is not a calibrated gain or early-stop predictor.

## Round 1: contract and transfer audit

The new same-scale 500K results share the `pcqm-fixed500k-dev50k-matched60-v4`
data, split, optimizer, schedule, raw-weight development selection, target
transform, seed and 234,360-step/29,998,080-presentation endpoint. Their
canonical traces pass SHA256 and exact-step alignment checks. Within this
group, reference-minus-candidate MAE is interpretable as a single-seed,
same-role mechanism comparison.

| GPTrans mechanism | 100K accepted gain | 500K selected gain | Contextual retention |
| --- | ---: | ---: | ---: |
| Noisy Nodes | +0.004840 eV | +0.001811 eV | 37.4% |
| Noisy Nodes + Pair Update Norm | +0.009382 eV | +0.002056 eV | 21.9% |
| Pair Update Norm alone | 100K exact match not established here | +0.001140 eV | not computed |

Retention is descriptive, **not a causal scale coefficient**. The 100K and
500K screens use different training rows, development cohorts, optimization
horizons and EMA-versus-raw selection. Both 100K winners selected their last
epoch. A row bootstrap would not estimate training-seed variance.

The isolated 500K gains are sub-additive as point estimates: `0.001140 +
0.001811 = 0.002951 eV`, versus `0.002056 eV` jointly. Thus the joint
improvement over Noisy Nodes alone is only about `0.000244 eV`; no robust
independent or synergistic effect is established by one seed.

## Round 2: matched 500K learning curves

| Joint versus GPTrans reference | Epoch 9 | Epoch 29 | Epoch 59 |
| --- | ---: | ---: | ---: |
| Matched development gain | +0.018467 | -0.000881 | +0.002067 eV |
| Online train metric difference | +0.013452 | +0.009128 | +0.008469 eV |

The joint development gain averages `+0.013116 eV` over epochs 0-9 but only
`+0.002024 eV` over epochs 50-59. Pair Norm alone and Noisy Nodes alone show
the same qualitative early-to-tail contraction. The joint terminal result is
therefore not a slow-start false negative. At the same time, the candidate's
online training metric remains lower. This is consistent with increased
training-fit benefit that does not carry proportionally to the development
cohort, **not proof of overfitting or representation collapse**: the online
train metric is accumulated during updates, not a fixed-subset re-evaluation.
The baseline's late improvement and candidate/reference schedules are visible
in `analysis.json`; the trace does not separate unique data size, optimization
horizon, target-cohort shift, EMA semantics or seed effects.

The accepted K1 and EdgeState 500K manifests and full 60-epoch traces were
also recovered as SHA-verified small JSON snapshots under `sources/`. They
share the GPTrans trace's exact 500K contract and step/sample axis:

| Gain over EdgeState at same step | Epoch 9 | Epoch 29 | Epoch 59 | Best-selected gain |
| --- | ---: | ---: | ---: | ---: |
| K1 | +0.039793 | +0.021420 | +0.005527 | +0.006489 eV |
| GPTrans-T | +0.017410 | +0.017854 | +0.004429 | +0.004481 eV |

K1's terminal online-train difference remains `+0.026180 eV` versus its
terminal development difference of `+0.005527 eV`. EdgeState therefore
**catches up late even within the fixed 500K contract**, before any full-data
change. This strengthens the horizon/sample-efficiency explanation for the
later rank reversal, but still does not identify whether more unique data,
more optimization, different regularization or the full training recipe
caused EdgeState to overtake them. The new snapshots are not yet canonical
RML replay-pool entries; they are accepted-source-backed postmortem evidence.

## Round 3: full-scale rank reversal

The matched 500K result ranked K1 `0.104860`, GPTrans `0.106868`, EdgeState
`0.111349 eV` on one internal-development role. The already accepted full
official-validation decision instead ranked EdgeState `0.099638`, K1
`0.106672`, GPTrans `0.109012 eV`; after its separately accepted continuation,
GPTrans reached `0.103791 eV`. These are within-role rankings, not valid
cross-role subtractions.

The full contracts were not equalized with 500K or across architectures.
K1/GPTrans initially received 20.0M presentations; selected/terminal
continuations brought them to approximately 30.14M and 40.27M. EdgeState's
selected zero-based epoch 30 implies about 31 full-data passes, or roughly
104.7M **nominal** presentations before any tail-batch adjustment. It also
used a different batch/optimizer/schedule; GPTrans used EMA full-scale while
the matched 500K run selected raw weights. EdgeState had seven subsequent
non-improving epochs, while GPTrans improved at every one of six continuation
evaluations and stopped at budget, not a plateau. K1's three continuation
evaluations were worse than its source after an LR restart, so that run did
not rescue K1 but does not prove every training schedule would fail.

Operationally, the selected EdgeState checkpoint still beats the selected
continued GPTrans by `0.004153 eV` and K1 by `0.007034 eV` on the consumed
official-validation role. It is therefore the safer accepted full-scale
reference. Whether GPTrans could close the remaining gap under equalized
compute, or whether K1's 500K advantage is mainly low-data inductive bias,
remains unidentified. Official validation was already used for selection;
these differences are not untouched leaderboard estimates.

## Minimum missing evidence, in cost order

1. **Zero GPU, partly completed here:** K1 and EdgeState matched-500K traces
   have been recovered, hashed and analyzed; their advantage does contract
   with horizon. Recover the exact PairToken/reference prefixes if durable
   artifacts still exist, and register the architecture traces as separate
   canonical RML replay entries before using them to calibrate a policy.
   Missing files must remain missing if unrecoverable.
2. **No immediate training:** freeze a genuinely shared internal-development
   role outside both train prefixes, and an explicit raw/EMA selection rule.
   The already reused 500K development role can support diagnostics but not an
   independent promotion claim. A new role must be declared before use; sealed
   official roles remain untouched.
   A local checkpoint inventory found retained 100K Noisy/Joint candidate
   files but not the accepted GPTrans-T 100K reference checkpoint whose
   acceptance SHA256 is `f4da386ae1e32f6953b645c0bdb8e208aaba1f1d7b8ebec132776bd63c22d6ab`.
   Therefore a frozen-weight 100K-to-500K-cohort paired inference was **not**
   run. Recover and hash-accept that exact reference first; a separately
   trained same-job control cannot silently replace it. Log any subsequent
   internal-role use, source indices, artifact hashes and measured native cost
   in a prospective RML action.
3. **Only if a full-scale allocation decision depends on it:** run a matched
   baseline/candidate pair under one fixed 500K recipe, identical seed,
   optimizer-step budget, LR trajectory, evaluation rows and both raw/EMA
   outputs. A second seed is needed to assess training variability. To
   isolate data diversity from optimization horizon, train the same pair on
   100K and 500K prefixes to the **same step checkpoints**, then compare
   fixed-prefix gains. Existing cross-scale scores cannot substitute for this
   factorial control. Predeclare hardware-native cost ceiling and a terminal
   decision gate before release; no such run is authorized here.
4. **Full-scale compute:** only after a bounded matched transfer survives,
   compare architectures under a separately budgeted full protocol. GPTrans
   currently lacks a demonstrated plateau, but another full continuation and
   official-role selection are costly and not automatically justified.

No retrospective early-stop threshold can be activated: the RML screening
backtest has no measured false-stop rate, false-promote rate, or native cost
savings for this cross-scale decision. A `B/16`/`B/4`/`B/2` checkpoint is an
observation schedule, not yet an automatic stop policy.

Machine evidence: `analysis.json` generated by `analyze.py`. The script reads
only accepted JSON traces/contracts, checks four GPTrans plus two architecture
500K trace hashes and comparability fields, and records the full source hashes
used for budget accounting. The prior frozen-readout result on branch commit
`4293763f` is contextual and does not change this decision.
