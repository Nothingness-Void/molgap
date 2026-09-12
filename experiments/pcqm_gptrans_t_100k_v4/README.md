# GPTrans-T 100K V4 reference

This experiment trains the immutable GPTrans-T reference once on the accepted
official-train-derived PCQM4Mv2 100K/50K identity. It exists to support later
V4 comparisons without rerunning a baseline per candidate or platform.

- Scientific contract: [`training_contract.json`](training_contract.json)
- Protocol and limits: [`protocol.md`](protocol.md)
- Authorization boundary: [`launch_decision.md`](launch_decision.md)
- Remote launch state: [`STATUS.md`](STATUS.md)
- Mechanical acceptance: [`accept_result.py`](accept_result.py)

Official validation, test-dev, and test-challenge are unavailable to this
experiment.
