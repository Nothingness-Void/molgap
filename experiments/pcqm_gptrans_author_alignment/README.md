# GPTrans author/local attribution matrix

The [prelaunch decision](decision.md) records the evidence boundary. The
[protocol](protocol.md) separates accepted input preflight from causal
training comparisons. The CPU real-graph input check is an infrastructure
gate, not a model result. Its executable is `kaggle_cpu/run.py`; the historical
synthetic and author-Cython parity result remains in
`../pcqm_gptrans_parity_cpu/decision.md`.

Submission and acceptance status is recorded in [STATUS](STATUS.md). No
accuracy effect is claimed before a separately frozen paired training arm
passes its own prelaunch and terminal gates.
