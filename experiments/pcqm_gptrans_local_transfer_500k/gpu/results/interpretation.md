# Equal-update500K local-stream transfer — terminal interpretation

On 2026-10-07, physical Kaggle2 kernel137461712 version1 was independently
accepted from its77 manifest-bound outputs. Full trace, source/receipt/config,
initialization, fixed rows and targets, runtime qualification, checkpoint hashes,
observed roles and allocation costs passed. No model execution occurred locally.
Authorities: [saved-output acceptance](acceptance.json),
[strict comparison](../scale_ema/results/comparison_readiness.json), and
[actual Replay proof](closure.json).

## Result: retained architecture benefit at500K

| Primary EMA999 endpoint | Parameters | Best rung, zero-based | Internal50K Gap MAE, eV |
|---|---:|---:|---:|
| Retained G1 control | 5,246,817 | 59 | 0.1152574413 |
| G1 + uncapped real-bond local stream | 5,871,201 | 59 | 0.1096073005 |

Selected gain was0.0056501408 eV, approximately4.90% of reference MAE, with
11.90% additional parameters. Candidate-minus-reference paired-row95% bootstrap
interval was[-0.0061879503,-0.0051088837] eV;53.59% of rows improved. The frozen
0.001 directional gate and historical0.003 material criterion were both passed.
Row bootstrap measures these saved molecules, not training-seed variation.

The benefit did shrink during optimization, but did not disappear:

| Optimizer steps | Presentations | Reference EMA MAE | Local EMA MAE | Gain, eV |
|---:|---:|---:|---:|---:|
| 7,810 | 999,680 | 0.163757995 | 0.149181068 | 0.014576927 |
| 15,620 | 1,999,360 | 0.137562007 | 0.127079353 | 0.010482654 |
| 23,430 | 2,999,040 | 0.125631869 | 0.118481718 | 0.007150151 |
| 31,240 | 3,998,720 | 0.119324863 | 0.112896807 | 0.006428055 |
| 39,050 | 4,998,400 | 0.116116837 | 0.110265352 | 0.005851485 |
| 46,860 | 5,998,080 | 0.115257442 | 0.109607294 | 0.005650148 |

The final-ten-rung mean gain was0.0057638347 eV. Final live development MAE
was0.1095591411 versus the control's0.1152308583, so the retained advantage was
not solely an EMA selection lag. The small difference between saved-prediction
MAE and trace MAE follows their double-precision versus producer reductions.
Training trace values are normalized-loss units; do not subtract them from
development MAE in eV to estimate an overfitting gap.

The earlier100K local screen improved0.0018744017 eV but failed its historical
gate. This500K comparison supports useful real-bond information after actual
expanded-data optimization. It does not retroactively change the100K decision.
Each gain compares internally matched arms; absolute100K and500K MAEs were not
subtracted. This is evidence against universal small-data-only benefit, not a
guarantee that the module will improve every500K/full recipe.

## Localization and causal limits

Post-hoc reference-error quintiles had candidate-minus-reference deltas
+0.023132,+0.006728,-0.006460,-0.017662,-0.033988 eV. The net benefit was stronger
on molecules that the reference predicted poorly, with losses among its easiest
rows. These target-derived groups are diagnostic, not deployable routing or
identified chemical subspaces; no chemistry/scaffold enrichment was measured.

The only trained architecture intervention was the original64-channel real-bond
stream before each12 GPA blocks. Improved explicit local communication is
consistent with the result. Width/parameter regularization, information-flow
amplitude and specific layer responsibility were not independently separated
by this training comparison. The failed cap experiment was not reopened.

## Cost and exposure

The notebook ran14,447.540863 seconds:4.0132 wall hours and8.0264 allocated T4
hours. One of two allocated devices trained; idle allocation was counted, not
hidden. All60 rung timings summed to10,011.943413 seconds for training including
loader/H2D and4,263.310270 seconds for evaluation, approximately69.3% and29.5%
of notebook wall time. Peak allocated memory was1,213,943,296 bytes. The bounded
four-update operator trace was retained for separately scoped compute analysis.
No matched detailed timer exists for the retained control, so no causal speed
ratio was inferred from different notebook envelopes.

The endpoint was46,860 updates /5,998,080 presentations: about12 passes, not60
full500K epochs. Both endpoints selected the last rung. Complete convergence,
later equal-exposure retention and full-scale ranking remain unresolved; the
result does not authorize extending only the candidate or changing its schedule.
This internal50K role cannot be compared directly with official-validation
EdgeState numbers. No official validation/test-dev/challenge was accessed.

## RML and disposition

The strict assessor and evidence-bound validator returned STRICT_CAUSAL with
no blockers. RML finalization was valid, and an idempotent retry returned
ALREADY_FINALIZED. The rebuilt Replay pool actually admitted both candidate
and retained control with capability=complete and identical comparability keys;
the new terminal outcome was POSITIVE_UNDER_CONTRACT. The old control's original
noncausal comparison and outcome remained unchanged.

The first real closure exposed a metadata adapter gap: native role/cost files
did not themselves contain RML event arrays. Separate source-SHA-bound event
and terminal-acceptance projections repaired that translation without editing
the remote data, recipe, thresholds or validator. Nineteen focused regressions
passed. Raw saved acceptance stayed separate from terminal interpretation.

Retain this candidate as a positive same-budget500K research result. The primary
remaining question is later-exposure retention, not another width/cap/addon grid.
Only the protocol's separately bound NO_TRAIN checkpoint probes may use its
reserved diagnostic allowance; their parent hashes and own records must first
be frozen. No probe or successor training was submitted during this acceptance.
Extra seeds, full training, desktop custody and protected evaluation were not
released. Monitor event custody was closed after interpretation.
