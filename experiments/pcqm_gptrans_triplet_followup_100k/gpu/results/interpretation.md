# Triplet follow-up: partial terminal interpretation

On 2026-10-13, exact Kaggle2 kernel 138367521/version 1 outputs were reconciled
and retrieved. The parent ended ERROR; this did not invalidate its complete
independent child. The existing native acceptance and RML transaction were
reused, without local model execution or baseline retraining.

| Arm | Saved-output result | Decision |
|---|---|---|
| Triplet-return drop p=0.1 | 0.1416272528 eV, selected epoch 44, 5,880,961 parameters | No nomination; worse than the immutable aggregate reference |
| Channelwise relation product | Failed initialization qualification before model forward or training | Infrastructure-only; scientific hypothesis not evaluated |

## Complete arm

The immutable aggregate reference gave 0.1393084501 eV on the identical 50,000
internal-development rows. Drop-minus-reference was +0.0023188027 eV;
the paired-row bootstrap 95% interval was [+0.0014419205, +0.0032298030] eV.
This interval describes saved rows, not training-seed uncertainty. The frozen
0.001 eV nomination gate was not met. The existing terminal owner retained its
conservative `INCONCLUSIVE` outcome; this record does not replace it with a
claim of universal architectural inferiority.

All 60 passes, 46,860 optimizer steps and 5,998,080 sample presentations were
verified. Hash checks covered selected model, saved predictions, last checkpoint,
six checkpoint chunks, frozen metadata and both trace forms. The native saved
prediction MAE was independently recomputed. Strict comparison acceptance and
actual replay-pool admission were verified for the complete arm only.

| Zero-based epoch | Reference train | Drop train | Reference EMA dev | Drop EMA dev |
|---|---:|---:|---:|---:|
| 9 | 0.176594 | 0.179184 | 0.173937 | 0.175177 |
| 19 | 0.140243 | 0.142516 | 0.148607 | 0.150688 |
| 29 | 0.116696 | 0.119149 | 0.141055 | 0.143545 |
| 39 | 0.094880 | 0.096878 | 0.139321 | 0.141972 |
| 59 | 0.071324 | 0.073902 | 0.139945 | 0.142216 |

The intervention raised both training and development error across these
matched later checkpoints. It did not cure late development degradation:
the drop arm's final EMA dev was 0.142216 eV and final live dev 0.142203 eV.
This supports closing this precise branch-drop intervention rather than
extending its horizon or claiming an EMA-only selection failure. It does not
establish that all forms of regularization are ineffective.

## Incomplete arm and cost

The [failure decision](../degree_local_triplet_product_ema999/failure_results/decision.md)
owns the exact stack, limits of the diagnosis and role semantics. A cross-runtime
random-initialization regeneration check is a plausible issue, not yet a proved
failed clause. No architecture score or training Replay admission is fabricated.

The physical allocation consumed 3.565514 wall hours and 7.131028 T4 device-hours.
Cost attribution retains one half per declared arm, including the failed arm's
idle reserved device; it does not claim that both devices trained throughout.
The accepted drop arm averaged 211.757 seconds per epoch.

No retry, extra seed, new training, 500K inference or full-scale promotion was
released. The deferred full-role action remains unexecuted. The old aggregate
reference remains preferred within this comparison contract.
