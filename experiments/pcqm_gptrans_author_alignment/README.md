# GPTrans author/local attribution matrix

The [prelaunch decision](decision.md) records the evidence boundary. The
[protocol](protocol.md) separates accepted input preflight from causal
training comparisons. The CPU real-graph input check is an infrastructure
gate, not a model result. Its executable is `kaggle_cpu/run.py`; the historical
synthetic and author-Cython parity result remains in
`../pcqm_gptrans_parity_cpu/decision.md`.

The separate real-input initialization-scale diagnostic uses
`kaggle_initial_cpu/run.py` and the same accepted fixed train rows plus the
frozen GPTrans seed-42 initial weights. It reads no labels and performs no
training, inference, or protected-role evaluation.

Submission and acceptance status is recorded in [STATUS](STATUS.md). No
accuracy effect is claimed before a separately frozen paired training arm
passes its own prelaunch and terminal gates.

The separately authorized [G1/G2 dual-arm protocol](dual_arm_protocol.md)
freezes degree-only initialization scaling and parameter-free chemical path
input. Its CPU preparation uses `stage_preparation.py` and
`kaggle_prepare/run.py`; this stage never releases GPU training by itself.
