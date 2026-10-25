# K1 100K to 500K attribution — 2026-10-25

Scope: desktop-owned evidence review and a fixed local CPU diagnostic, bounded
by 05:41:35–07:41:35 UTC. This is not a model promotion or another training run.
Historical server records are pinned local evidence, not server runtime custody.

## Findings that change the decision

### 1. Recent variant gains do not establish gains over original K1

| Accepted 100K question | Reference / candidate MAE, eV | What the pair actually tests |
|---|---:|---|
| MsgNorm fixed / learned scale | 0.146133172 / 0.144769128 | Learning the scale within MsgNorm; +1.364 meV, below the 3 meV gate |
| Hop mixed / ordered shells | 0.142960442 / 0.141844653 | Shell ordering with equal content; +1.116 meV, below the gate |
| BatchRenorm scale / scale+mean | See owning decision | Adding mean correction; 1.646 meV worse, and 14.037 meV worse after fixed Clean-BN |
| Original V4 K1 | 0.141373634 | Historical context only for the above independently trained pairs |

The first two pairs do not contain original K1 as a scientific comparator.
Calling them large absolute K1 improvements misstates the experiment. The
same-row four-expert average did improve a two-expert average by 3.101 meV,
but changes model count and cost. It is evidence of output complementarity,
not evidence that inserting all modules into one K1 will add their gains.

Authorities: [MsgNorm](../pcqm_k1_message_norm_100k/terminal_decision.md),
[shells](../pcqm_k1_hop_shell_100k/terminal_decision.md),
[BatchRenorm](../pcqm_k1_batch_renorm_100k/final_attribution.md),
[fixed fusion](../pcqm_k1_hop_fusion_qualification/terminal_decision.md).

### 2. The strongest 100K package was not transferred intact

The pretrained consistency pair achieved 0.138265848 eV without the teacher
and 0.136781212 eV with it. The same-job teacher increment was 1.484636 meV,
row interval [0.759455, 2.196867]. Both arms shared the pretrained initialization;
this is not an isolated pretraining benefit.

The composed 500K K1 deliberately omitted the teacher. It used a 10-pass,
100K pretrained backbone with a reset head, then 60 downstream passes. That
pretraining covers 20% of downstream membership and 3.33% of its presentations.
Those counts do not prove that longer or broader pretraining would help.
Comparing the strongest teacher-equipped 100K package with teacher-free 500K
cannot identify a pure scale effect or a missing-capacity bottleneck.

The composed K1's 0.104904077 versus historical K1's 0.104859870 is essentially
unchanged descriptively (+0.044 meV error). The companion GPTrans improved from
historical 0.106867528 to selected EMA 0.101887690. That ranking reversal includes
a substantially better GPTrans endpoint; it is not evidence that K1 itself lost
several meV of accuracy as data grew.

The [historical source ledger](../pcqm_k1_complete_local_audit/historical_evidence.md)
binds the original D10, D13–D15 contracts and decisions, including retired owner
refs. LIVE, EMA and Clean-BN are different output states and remain separate.

### 3. PairToken has direct evidence of weak portability, before retraining

Its 100K gain was 3.043577 meV, only 0.043577 above the material gate. Applying
the same frozen 100K weights to the 500K development cohort reduced that gain
to about 0.416 meV. The loss of gain is 2.627 meV, paired-row interval
[-3.805, -1.441] meV. No extra training was needed to observe this loss.

This identifies cohort-dependent gain for those frozen predictors. It does not
isolate a chemical cause or measure seed variability. Simple atom-count and
conjugation reweighting explained only small portions in the retained audit;
the within-group effect remained. A universal distribution-shift diagnosis is
stronger than the evidence supports.

The early 500K report used historical scalar K1 0.104859870 and reported a
0.556 meV gain. Its later recovered matched reference is 0.103605195, against
PairToken 0.104303729: **0.699 meV worse**, row interval [0.139, 1.253] meV.
Keep the dated early report; use the recovered matched comparison for inference.
The two reference values are not interchangeable.

Authority: the same [pinned historical ledger](../pcqm_k1_complete_local_audit/historical_evidence.md),
its exact-reference and frozen cross-cohort sources. The older
[scale summary](../pcqm_scale_transfer_attribution/decision.md) predates that
reference recovery and must not be treated as the latest matched comparison.

### 4. The curves do not support a universal early learning plateau

Original 100K K1 selected the last of 40 passes, at 31,240 updates and 3,998,720
presentations; its last ten passes still improved by 1.950 meV. Original 500K K1
selected pass 49 of 60, at roughly 24.5M presentations, not the beginning of
training. Its final point was 1.062 meV worse than its selected point.

Under the matched 500K contract, K1's advantage over EdgeState shrank from
39.793 meV at zero-based epoch 9 to 5.527 meV at epoch 59. Other models catching
up and K1's late regression differ from “K1 stops learning immediately.”
Epochs at 100K and 500K differ by roughly five times in presentations, and their
development members differ. Exposure-aligned descriptive curves do not by
themselves identify a causal scale law.

Authority: original V4 curves, the complete historical ledger and the accepted
[clean-fit diagnostic](../pcqm_k1_clean_fit_generalization_500k/terminal_decision.md).

### 5. BN is a measurable penalty, but does not explain most late regression

At the same composed K1 pass-49 weights, TRAIN-only Clean-BN lowers development
MAE from 0.104904075 to 0.103658948: 1.245127 meV, interval [1.107800, 1.379097].
This is a fixed-weight output-state effect. It accounts arithmetically for about
41% of the gap to the selected GPTrans EMA endpoint, leaving about 1.771 meV.
It does not establish a training-time explanation or an adopted new model.

From pass 49 to 60, consistency weights became 0.787332 meV worse even after
the same Clean-BN procedure, interval [0.629833, 0.950306]. That is 95.95% of
the raw late error increase. This percentage is an error decomposition, not
the proportion proved to be overfitting. On the fixed calibrated TRAIN cohort,
error nevertheless improved by about 0.538 meV. The observed late tradeoff is
better sampled training fit without better development performance.

Authorities: [BN row attribution](../pcqm_k1_bn_row_attribution/analysis_zh.md),
[clean-fit attribution](../pcqm_k1_clean_fit_generalization_500k/attribution.md).

## What the new retained-tail diagnostic adds

The [prospective protocol](protocol.md) fixes five retained LIVE states: the
pass-49 initialization, and selected pass 15/final pass 20 from both tail arms.
Each is evaluated with original BN and independently TRAIN-calibrated BN, on
the same 16,384 TRAIN members and 50,000 development members. The exact frozen
factory, checkpoints, graph hashes and original prediction guards are reused.

The first attempt failed before inference on an import-order guard. Its label
reads, measured wall time and missing CPU timer are retained truthfully. The
separately prospective [retry](attempt_002/protocol.md) owns the completed
scientific matrix and its [terminal interpretation](attempt_002/terminal_decision.md).
These diagnostics cannot repair the original training run's missing strict
target evidence or convert it to replay-ready by themselves.

## Remaining identification limits

- No seed replication establishes a variance band for tiny 100K wins.
- Most recent module pairs compare variants, not original K1; their absolute
  benefit is unmeasured under a matched reference contract.
- The tail experiment resets optimizer state and changes the parent loss/LR
  recipe. Its within-pair contrast isolates weight decay; its contrast against
  initialization cannot separately blame the reset, single forward or LR.
- Calibrated TRAIN errors use the same members used to fit BN statistics.
  They describe in-sample fit, not independent calibration generalization.
- Low effective rank, active module dependence and failed width experiments
  do not establish a K1 architectural ceiling or the correct larger dimension.
- Repeatedly consumed development and conditional row intervals do not provide
  independent test performance or training-seed confidence.

## Operational conclusion

Do not stack the recent small variant gains into another expensive 500K model,
repeat this tail weight-decay grid, or add epochs solely because K1 appears to
converge early. Existing evidence already rejects those justifications.

The next useful training question should compare identical 500K output-state,
exposure and runtime contracts while isolating one still-unidentified recipe
choice. A single-forward, fixed Clean-BN comparison of scratch initialization
against the retained 100K-pretrained backbone would isolate the claimed
pretraining transfer, while avoiding two-forward cost and raw/EMA confusion.
This is a proposed discriminator, not a proven better model or a launched job.
Confirm it is not already answered by a strict owner before freezing it; no
automatic retry, promotion, new stopping rule or full-data run is authorized
by this report. The two-hour work prioritizes the observed attribution and
concrete acceptance repairs over another unidentifiable training package.
