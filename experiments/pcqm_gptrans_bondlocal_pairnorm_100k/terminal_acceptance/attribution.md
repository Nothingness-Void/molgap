# BondLocal plus fresh pair-update normalization100K

2026-10-27 JST. This same-job intervention normalizes each fresh32-channel
pair update before persistent addition. It retains degree-scaled BondLocal,
the transported initialization, parameter count and frozen training recipe.
It does not normalize the accumulated pair state or isolate BondLocal's effect.

Selected EMA MAE changes from0.142861308 to0.139718318eV:
candidate-minus-reference=-3.142991meV, paired-row95%CI[-4.056945,-2.271359]meV.
The frozen3meV/negative-upper-bound gate passes. Its margin above the material
threshold is0.142991meV; the interval is conditional on this seed and selected
models, not measured seed variation or independent-test uncertainty.

Both arms complete60passes/46860updates/5998080presentations. Retained EMA
observations show a persistent development advantage, with diminishing late gain:

| Pass | BondLocal EMA/eV | Pair-update-norm EMA/eV |
| --- | ---: | ---: |
| 10 | 0.178857952 | 0.171231657 |
| 20 | 0.153451160 | 0.148784146 |
| 30 | 0.146084994 | 0.142400533 |
| 40 | 0.143679023 | 0.139718309 |
| 50 | 0.143024892 | 0.140440285 |
| 60 | 0.142951921 | 0.140582591 |

Reference selects pass52; candidate selects40 and worsens0.864meV by60.
Its online normalized TRAIN objective still falls from0.074206 at40 to0.055643
at60. This describes divergence between two recorded quantities; stochastic
TRAIN objective and development EMA MAE are different predictors/metrics.
It does not uniquely identify overfitting, a capacity ceiling or a cause of the
normalization benefit. No activation/gradient statistics or controlled amplitude
intervention establish that unstable message scales caused the reference error.

The intervention has a useful conditional100K signal. Historical standalone
PairNorm500K missed its own gate; this joint result neither reverses that decision
nor proves additive gains or500K transfer. No model adoption, automatic tuning,
500K/full training or protected-role evaluation is released here.

Physical entry allocation is10282.84574055seconds on two Tesla T4s:
5.712692078device-hours including preparation, preflight, evaluation and assigned
idle time. The narrower Python ledger is included in that window, not another
charge. Arm process walls are9870.283/9911.765seconds; they are not separate
full-device charges or compute-matched speed measurements. Queue, pre-entry,
release and device busy time remain unknown.

This uses retained traces/predictions and the existing paired analysis only.
Canonical strict qualification and replay status are owned by the terminal records.
