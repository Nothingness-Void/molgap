# Two-family500K transfer: substantial gains, no incumbent nomination — 2026-10-27

Kaggle2 kernel140916841/version3 completed both frozen exact-step endpoints.
Independent saved-output and native-state qualification passed for both arms.
The resumed traces retained their original prefixes. Source, rows, targets,
checkpoints, native repeat/next-step proofs and role audits were retained.
Only train500K and Common-Dev were consumed; protected roles were untouched.

| Family | Parameters | Source100K selected MAE |500K selected/final MAE | Selected gain |
|---|---:|---:|---:|---:|
|PPGT mean|2,577,729|0.2004566171|0.1714706253|0.0289859918|
|ESA count|5,015,745|0.2004349773|0.1376542035|0.0627807738|

Both comparisons exactly aligned all50,000 saved Common-Dev IDs and labels.
Within each family, initialization, FP32/BS128, LR-by-update clock and total
presentations were preserved while training membership expanded. PPGT completed
15,620 updates/1,999,360 presentations; ESA completed46,860/5,998,080. The two
families had different learning rates/exposures and were not a strict causal
head-to-head architecture comparison.

Paired selected-prediction row bootstrap gain95% intervals were
[0.0275334001,0.0305394653]eV for PPGT and
[0.0612564689,0.0644647702]eV for ESA. They described these reused rows, not
training seeds or final-live confidence. Final-five matched-curve mean gains
were0.0271612614/0.0645262837eV. The500K trajectories did not exhibit the old
100K-only gain collapse in this comparison. Larger training diversity therefore
remained a useful explanation for the previous weak small-data performance;
this noncausal transfer study did not isolate it from all execution/history
uncertainty.

ESA changed only0.0001226664eV across its final five evaluations and was close
to a plateau under this budget. PPGT improved0.0051271915eV across those points
and remained less settled, but this did not authorize extension. Online training
metrics were normalized-loss measurements and must not be subtracted directly
from eV development MAE.

The frozen rescue gate was not fully qualified: neither source100K run retained
an independently accepted final-live prediction payload, so no final-live row
confidence was reconstructed. No qualified strong K1/GPTrans comparator was
prospectively frozen for this Common-Dev role. In particular the accepted
GPTrans local500K record used different row/target hashes, and its0.1096073eV
was not a directly comparable score. No incumbent replacement, full-training
handoff, new seed, baseline inference or scientific successor was released.
The terminal scientific disposition was INCONCLUSIVE for nomination, while
retaining both large positive transfer observations for future decisions.

The three deduplicated R3 allocation segments measured19,909.027493889 wall
seconds (5.5302854 session hours),39,818.054987778 physical device seconds
(11.0605708 T4-hours), within the32,370-second aggregate ceiling. Overlapping
bootstrap, worker and shared-ledger copies were not summed. Earlier startup
faults retained their separate receipts/costs. Idle allocated ESA time while
PPGT finished was charged; these were not both busy-GPU hours. Actual billing,
provisioning-before-Python and queue time were unobserved.

The submission helper had written an expanded JSON receipt representation.
A canonical, semantically identical copy passed the existing receipt reader;
[normalization evidence](receipt_encoding_normalization.json) bound both hashes
and preserved the original bytes. No scientific identity was changed.

[Summary and matched curves](summary.json), [saved-output acceptance](output_acceptance.json)
and [native qualification](native_acceptance.json) own numeric evidence.
RML terminal/replay status is reported separately; neither execution completion
nor this positive descriptive contrast established strict causality or Replay.
