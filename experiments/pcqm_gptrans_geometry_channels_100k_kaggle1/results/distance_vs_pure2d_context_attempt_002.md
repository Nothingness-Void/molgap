# Saved-prediction context for the distance-only arm

The retained `distance_only` predictions were compared with two previously
accepted pure-2D GPTrans-T references. All three files contain the same 50,000
ordered internal-development source indices (100,000–149,999) and exactly equal
target tensors. Each prediction file's SHA256 matches its owning acceptance
record. The analysis loaded saved predictions only; it ran no model and read no
protected evaluation role. Exact inputs and statistics are in
[`distance_vs_pure2d_context_attempt_002.json`](distance_vs_pure2d_context_attempt_002.json).

| Existing pure-2D reference | Reference MAE | Distance-only MAE | Distance minus reference | Paired row-bootstrap 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Kaggle1 T4 | 0.1560144881 eV | 0.1519791187 eV | −4.035 meV | [−4.997, −3.074] meV |
| Kaggle2 P100 V4 | 0.1566272043 eV | 0.1519791187 eV | −4.648 meV | [−5.568, −3.692] meV |

This is a **contextual positive signal for the distance channel**, not the
frozen scientific decision of this experiment. The references came from
separate training jobs; the second used a different accelerator. The 10,000
paired row-bootstrap draws (seed 42) quantify row uncertainty conditional on
those fitted checkpoints and do not estimate training-seed variation. The
current V5 strict mechanism contract does not allow `feature_identity` as an
intervention, and the geometry job's trace lacks live-development observations.
The distance arm therefore has no strict cross-experiment promotion claim or
replay-ready status from this analysis.

The pre-registered angle increment remains negative by
[`decision_attempt_002.md`](../decision_attempt_002.md). Any future attempt to
adopt distance geometry requires a separately frozen scientific question and
an explicit comparison contract; these saved-prediction numbers alone do not
release training or a protected-role read.
